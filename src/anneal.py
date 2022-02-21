import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np
import sklearn as sk

import train_env
import model

# Used to calculate the total hamiltonian of a certain problem
def total_hamiltonian(mu, sigma, C_i, C_ij):
    ''' Derived from Eq. 9 in QAML-Z paper (ZLokapa et al.)
    '''
    ham = np.sum(-C_i + np.sum(np.einsum('ij, jk', np.triu(C_ij, k=1), mu))) * sigma
    ham += np.sum(np.triu(C_ij, k=1)) * pow(sigma, 2)
    
    return ham

# Makes the h and J np arrays for use in creating the bqm and networkx graph
def make_h_J(C_i, C_ij, mu, sigma):
    h = 2 * sigma * (np.einsum('ij, jk', C_ij, mu) - C_i)
    J = 2 * np.triu(C_ij, k=1) * pow(sigma, 2)
    
    return h, J

# Independent function for simplifying problem
# -> should hold a couple simple pruning methods
def default_prune(J, cutoff_percentile):
    rows, cols = np.shape(J)
    sign_J = np.sign(J)
    J = np.abs(np.ndarray.flatten(J))
    np.where(J > cutoff_percentile, J, 0)

    return np.broadcast_to(J * sign_J, (rows, cols))

# makes a dwave bqm and corresponding networkx graph
def make_bqm(h, J, fix_var):
    bqm_nx = nx.from_numpy_matrix(J)
    atrr_arr = np.repeat(np.array(['h_bias']), np.size(h))
    atrr_arr = np.column_stack(atrr_arr, h)
    for val in np.nditer(atrr_arr):
        val = dict(val)
    h_dict = np.column_stack(np.arange(np.size(h)), atrr_arr)
    bqm_nx.add_nodes_from(h_dict)
    
    bqm = dimod.from_networkx_graph(bqm_nx, vartype='SPIN', node_attribute_name='h_bias', edge_attribute_name='J_bias')
    if fix_var:
        fixed_dict = dimod.roof_duality.fix_variables(bqm)        # Consider using the more aggressive form of fixing
        for i in fixed_dict.keys():
            bqm.fix_variable(i, fixed_dict[i])

    return bqm, bqm_nx, fixed_dict

# Independent function for quantum error correction
# -> should hold a single correction scheme to start (can always add)
def default_qac(h, J, fix_var, C=4):
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
def default_copy(h, J, fix_var, C=4):
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
def dwave_connect(config, iter, sampler, bqm, bqm_nx, A_adj, A, anneal_time):
    num_nodes = bqm.num_variables
    qaresults = np.zeros((config.ngauges[iter]*config.nread, num_nodes))
    for g in range(config.ngauges[iter]):
        a = np.sign(np.random.rand(num_nodes) - 0.5)
        embedding = minorminer.find_embedding(bqm_nx, A)
        th, tJ = dw.embedding.embed_ising(nx.classes.function.get_node_attributes(bqm_nx, 'h_bias'), nx.classes.function.get_edge_attributes(bqm_nx, 'J_bias'), embedding, A_adj)
        th, tJ = scale_weights(th, tJ, config.strengths[iter])

        qaresult = sampler.sample_ising(th, tJ, num_reads = config.nread, annealing_time=anneal_time, answer_mode='raw')
        unembed_qaresult = dw.embedding.unembed_sampleset(qaresult, embedding, bqm)

        for i in range(len(unembed_qaresult.record.sample)):
            unembed_qaresult.record.sample[i, :] = unembed_qaresult.record.sample[i, :] * a
        qaresults[g*config.nread:(g+1)*config.nread] = unembed_qaresult.record.sample

    return qaresults

# Does the fullstring stuff -> modifies results by undoing variable fixing
def full_string(qaresults, C_i, fix_vars, fixed_dict=None):
    '''
    TODO: see if can speed up the process
    '''
    full_strings = np.zeros((len(qaresults), len(C_i)))
    if fix_vars:
        j = 0
        for i in range(len(C_i)):
            if i in fixed_dict:
                full_strings[:, i] = 2*fixed_dict[i] - 1
            else:
                full_strings[:, i] = qaresults[:, j]  
                j += 1
    else:
        full_strings = qaresults

    return full_strings

# Derives the energies obtained from the annealing
def en_energy(s, sigma, qaresults, C_i, C_ij, mu):
    '''
    TODO: see if can speed up the process with numpy functions
    '''
    en_energies = np.zeros(len(qaresults))
    s[np.where(s > 1)] = 1.0
    s[np.where(s < -1)] = -1.0
    bits = len(s[0])
    for i in range(bits):
        en_energies += 2*s[:, i]*(-sigma*C_i[i])
        for j in range(bits):
            if j > i:
                en_energies += 2*s[:, i]*s[:, j]*pow(sigma, 2)*C_ij[i][j]
            en_energies += 2*s[:, i]*sigma*C_ij[i][j] * mu[j]

    return en_energies

# Picks out the unique energies out of the ones derived from en_energy
def unique_energy(en_energies, config, iter):
    '''
    TODO: see if can speed up the process with numpy functions
    '''
    unique_energies, unique_indices = np.unique(en_energies, return_index=True)
    ground_energy = np.amin(unique_energies)
    if ground_energy < 0:
        threshold_energy = (1 - config.energy_fractions[iter]) * ground_energy
    else:
        threshold_energy = (1 + config.energy_fractions[iter]) * ground_energy
    lowest = np.where(unique_energies < threshold_energy)
    unique_indices = unique_indices[lowest]
    if len(unique_indices) > config.max_states[iter]:
        sorted_indices = np.argsort(en_energies[unique_indices])[-config.max_states[iter]:]
        unique_indices = unique_indices[sorted_indices]
    
    return unique_indices

def anneal(config, iter, C_i, C_ij, mu, sigma, A_adj, A, sampler, anneal_time=5):
    ''' Controls the annealing and energy selection process.

        TODO: move sigma to config?; change C_i and C_ij to env; change param ordering in function call;
            remove A_adj and A b/c both under sampler

        Called from train() multiple times. Each call to anneal performs a training step
        on the D-Wave hardware.

        Parameters:
        - C_i          The intermediate form of your input data that defines the graph nodes.
        - C_ij         The intermediate form of your input data that defines the graph edges.
        - mu           Vector of expected values of qubit spins.
        - sigma        The training rate in range [0..1]
        - config       Configuration struct (hold many useful config params)
        - iter         Declares the current iteration within the training method.
                       This is needed because some configuration parameters are per-iteration arrays.
        - A_adj        The adjacency <something> of the annealer.
        - A            The networkx graph of the annealer.
        - sampler      The connection to D-Wave.
        - anneal_time  How long the annealing occurs. The longer you anneal, the better the
                       result. But the longer you anneal, the higher the chance for decoherence
                       (which yields garbage results). Units = microseconds.
    '''
    prune_vars = config.prune_vars
    cutoff = config.cutoff
    encode_vars = config.encode_vars
    fix_vars = config.fix_vars

    h, J = make_h_J(C_i, C_ij, mu, sigma)
    if not prune_vars is None:
        J = prune_vars(J, cutoff)
    if encode_vars is None:
        bqm, bqm_nx, fixed_dict = make_bqm(h, J, fix_vars)
    else:
        orig_len = np.size(h)
        bqm, bqm_nx, fixed_dict = encode_vars(h, J, fix_vars)

    qaresults = dwave_connect(config, iter, sampler, bqm, bqm_nx, A_adj, A, anneal_time)

    full_strings = full_string(qaresults, C_i, fix_vars, fixed_dict=fixed_dict)
    en_energies = en_energy(full_strings, sigma, qaresults, C_i, C_ij, mu)
    unique_indices = unique_energy(en_energies, config, iter)
    final_answers = full_strings[unique_indices]
    
    return final_answers