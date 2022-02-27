import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np
import statistics as stat

import dwave.preprocessing.lower_bounds as dwplb
import dwave.embedding as dwe

# Makes the h and J np arrays for use in creating the bqm and networkx graph
def make_h_J(C_i, C_ij, mu, sigma):
    h = 2 * sigma * (np.einsum('ij, j', C_ij, mu) - C_i)
    J = 2 * np.triu(C_ij, k=1) * pow(sigma, 2)
    
    return h, J

# Independent function for simplifying problem
# -> should hold a couple simple pruning methods
def default_prune(J, cutoff_percentile):
    rows, cols = np.shape(J)
    sign_J = np.ndarray.flatten(np.sign(J))
    J = np.abs(np.ndarray.flatten(J))
    np.where(J > cutoff_percentile, J, 0)

    return np.reshape(J * sign_J, (rows, cols))

# makes a dwave bqm and corresponding networkx graph
def make_bqm(h, J, fix_var):
    bqm_nx = nx.from_numpy_matrix(J)
    attr_dict, h_list = {}, []
    for i in range(np.size(h)):
        attr_dict['h_bias'] = h[i]
        h_list.append((i, attr_dict['h_bias']))
    bqm_nx.add_nodes_from(h_list)
    
    bqm = dimod.from_networkx_graph(bqm_nx, vartype='SPIN', node_attribute_name='h_bias', edge_attribute_name='weight')
    fixed_dict = None
    if fix_var:
        lowerE, fixed_dict = dwplb.roof_duality(bqm, strict=True)
        if fixed_dict == {}:
            lowerE, fixed_dict = dwplb.roof_duality(bqm, strict=False)
        for i in fixed_dict.keys():
            bqm.fix_variable(i, fixed_dict[i])

    return bqm, bqm_nx, fixed_dict

# Independent function for quantum error correction
# -> should hold a single correction scheme to start (can always add)
def default_qac(h, J, fix_var, C):
    rows, cols = np.shape(J)
    con_J = J + J.T + np.max(np.ndarray.flatten(J))*np.eye(rows, cols)
    qa_J = np.zeros(C * (rows, cols))
    for i in np.arange(C):
        for j in np.arange(i, C):
            if i==j:
                qa_J[i*rows : (i+1)*rows, j*cols : (j+1)*cols] = J
            else:
                qa_J[i*rows : (i+1)*rows, j*cols : (j+1)*cols] = con_J
    h = np.repeat(h, C) * C

    return make_bqm(h, qa_J, fix_var)

# Used to compare/benchmark the performance of the error correction
def default_copy(h, J, fix_var, C):
    rows, cols = np.shape(J)
    cp_J = np.zeros(C * (rows, cols))
    for i in np.arange(C):
        cp_J[i*rows : (i+1)*rows, i*cols : (i+1)*cols] = J
    h = np.repeat(h, C)

    return make_bqm(h, cp_J, fix_var)

# adjust weights
def scale_weights(th, tJ, strength):
    for k in list(th.keys()):
        th[k] /= strength 
    for k in list(tJ.keys()):
        tJ[k] /= strength

    return th, tJ

# Does the actual DWave annealing and connecting
def dwave_connect(config, iter, sampler, bqm, bqm_nx, anneal_time):
    '''
        TODO: Determine if its possible to reuse the embedding from previous steps
    '''
    num_nodes = bqm.num_variables
    qaresults = np.zeros((config.ngauges[iter]*config.nread, num_nodes))
    for g in range(config.ngauges[iter]):
        a = np.sign(np.random.rand(num_nodes) - 0.5)
        if config.embedding is None:
            config.embedding = minorminer.find_embedding(bqm_nx, sampler.to_networkx_graph())
        th, tJ = dwe.embed_ising(nx.classes.function.get_node_attributes(bqm_nx, 'h_bias'), nx.classes.function.get_edge_attributes(bqm_nx, 'weight'), config.embedding, sampler.adjacency)
        th, tJ = scale_weights(th, tJ, config.strengths[iter])

        qaresult = sampler.sample_ising(th, tJ, num_reads=config.nread, annealing_time=anneal_time, answer_mode='raw')
        unembed_qaresult = dw.embedding.unembed_sampleset(qaresult, config.embedding, bqm)

        for i in range(len(unembed_qaresult.record.sample)):
            unembed_qaresult.record.sample[i, :] = unembed_qaresult.record.sample[i, :] * a
        qaresults[g*config.nread:(g+1)*config.nread] = unembed_qaresult.record.sample

    return qaresults

# Modifies results by undoing variable fixing
def unfix(qaresults, h_len, fixed_dict):
    spins = np.zeros((np.size(qaresults, 0), h_len))
    j = 0
    for i in range(h_len):
        if i in fixed_dict:
            spins[:, i] = fixed_dict[i]
        else:
            spins[:, i] = qaresults[:, j]
            j += 1

    return spins

# Modifies results by undoing variable encoding (qac method)
#   -> uses majority voting
def decode_qac(qaresults, h_len, orig_len, fix_vars, fixed_dict=None):
    if fix_vars:
        spins = unfix(qaresults, h_len, fixed_dict=fixed_dict)

    true_spins = np.zeros(np.size(qaresults, 0), orig_len)
    for i in range(orig_len):
        qubit_copies = []
        for j in range(np.size(spins, 0) / orig_len):
            qubit_copies.append(spins[j*orig_len])
        true_spins[i] = np.sign(stat.mean((qubit_copies)))
    true_spins[np.where(true_spins == 0)] = np.random.choice([-1, 1])

    return true_spins

# Modifies results by undoing variable encoding (copy method)
def decode_copy(qaresults, h_len, orig_len, fix_vars, fixed_dict=None):
    if fix_vars:
        spins = unfix(qaresults, h_len, fixed_dict=fixed_dict)
    
    multi_spins = np.zeros(np.size(spins, 0) / orig_len)       # will be an exact int, not a floating point
    for i in range(np.size(multi_spins)):
        multi_spins[i] = spins[:, i*orig_len : (i+1)*orig_len]

    return multi_spins

# Derives the energies obtained from the annealing
def energies(spins, sigma, qaresults, C_i, C_ij, mu):
    '''
        TODO: check ordering of einsum indecies; maybe add .T
    '''
    en_energies = np.zeros(np.size(qaresults, 0))
    np.sign(spins, spins)

    en_energies = -2 * np.einsum('ij, j', spins, C_i) * sigma
    en_energies += 2 * np.einsum('ij, j', np.einsum('ij, jk', spins, np.triu(C_ij, k=1)), spins[0][:]) * pow(sigma, 2)
    en_energies += 2 * np.einsum('ij, j', np.einsum('ij, jk', spins, C_ij), mu)

    return en_energies

# Derives the energies obtained from the annealing (qac method)
#   -> bc majority vote happens earlier, qac has the same structure as non-qac
def energies_qac(truespins, sigma, qaresults, C_i, C_ij, mu):
    return energies(truespins, sigma, qaresults, C_i, C_ij, mu)

# Derives the energies obtained from the annealing (copy method)
def energies_copy(multi_spins, sigma, qaresults, C_i, C_ij, mu):
    multi_energies = np.zeros(np.size(multi_spins, 0))
    for i in range(np.size(multi_spins, 0)):
        multi_energies[i] = energies(multi_spins[i], sigma, qaresults, C_i, C_ij, mu)

    return multi_energies

# Picks out the unique energies out of the ones derived from energies
def unique_energies(en_energies, energy_fraction, max_state):
    '''
        TODO:
    '''
    unique_energies, unique_indices = np.unique(en_energies, return_index=True)
    ground_energy = np.amin(unique_energies)
    if ground_energy < 0:
        threshold_energy = (1 - energy_fraction) * ground_energy
    else:
        threshold_energy = (1 + energy_fraction) * ground_energy
    lowest = np.where(unique_energies < threshold_energy)
    unique_indices = unique_indices[lowest]
    if np.size(unique_indices, 0) > max_state:
        sorted_indices = np.argsort(en_energies[unique_indices])[-max_state :]
        unique_indices = unique_indices[sorted_indices]
    
    return unique_indices

# Picks out the unique energies out of the ones derived from energies (qac method)
#   -> bc majority vote happens earlier, qac has the same structure as non-qac
def uniques_qac(en_energies, energy_fractions, max_states):
    return unique_energies(en_energies, energy_fractions, max_states)

# Picks out the unique energies out of the ones derived from energies (copy method)
def uniques_copy(multi_energies, energy_fractions, max_states):
    '''
        TODO:
    '''
    multi_uniques = np.zeros(np.size(multi_energies, 0))
    for i in range(np.size(multi_energies, 0)):
        multi_uniques[i] = energies(multi_energies[i], energy_fractions, max_states)

    return multi_uniques

def anneal(config, iter, env, mu):
    ''' Controls the annealing and energy selection process.

        TODO:

        Called from train() multiple times. Each call to anneal performs a training step
        on the D-Wave hardware.

        Parameters:
        - config       Configuration struct (hold many useful config params)
        - iter         Declares the current iteration within the training method.
        - mu           Vector of expected values of qubit spins.
                       This is needed because some configuration parameters are per-iteration arrays.
        - env          The TrainEnv variable that holds the environment (data) used for training.
        - anneal_time  How long the annealing occurs. The longer you anneal, the better the
                       result. But the longer you anneal, the higher the chance for decoherence
                       (which yields garbage results). Units = microseconds.
    '''
    h, J = make_h_J(env.C_i, env.C_ij, mu, pow(config.zoom_factor, iter))
    if config.prune_vars is not None:
        J = config.prune_vars(J, config.cutoff)
    if config.encode_vars is None:
        bqm, bqm_nx, fixed_dict = make_bqm(h, J, config.fix_vars)
        if config.fixed_dict is None:
            config.fixed_dict = fixed_dict
        elif config.fixed_dict.keys() == fixed_dict.keys():
            config.same_fixed_dict_counter += 1
        else:
            config.diff_fixed_dict_counter += 1
    else:
        orig_len = np.size(h)
        bqm, bqm_nx, fixed_dict = config.encode_vars(h, J, config.fix_vars, config.encoding_depth)

    qaresults = dwave_connect(config, iter, env.sampler, bqm, bqm_nx, config.anneal_time)

    if config.encode_vars is None and config.fix_vars is not None:
        spins = unfix(qaresults, np.size(h), fixed_dict=fixed_dict)
    elif config.encode_vars is not None:
        spins = config.decode_vars(qaresults, np.size(h), orig_len, config.fix_vars, fixed_dict=fixed_dict)
        en_energies = config.encoded_energies(spins, pow(config.zoom_factor, iter), qaresults, env.C_i, env.C_ij, mu)
        unique_indices = config.encoded_uniques(en_energies, config.energy_fractions[iter], config.max_states[iter])

        if len(np.shape(spins)) == 2:
            excited_states_arr = [spins[unique_indices]]
        elif len(np.shape(spins)) == 3:
            excited_states_arr = []
            for i in range(np.size(spins, 0)):
                excited_states_arr[i] = spins[i][unique_indices]
        
        return excited_states_arr
    else:
        spins = qaresults

    en_energies = energies(spins, pow(config.zoom_factor, iter), qaresults, env.C_i, env.C_ij, mu)
    unique_indices = unique_energies(en_energies, config.energy_fractions[iter], config.max_states[iter])    
    excited_states_arr = [spins[unique_indices]]
    
    return excited_states_arr