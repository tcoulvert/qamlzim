from unittest import skip
import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np
import os
import statistics as stat

import dwave.preprocessing.lower_bounds as dwplb
import dwave.embedding as dwe

# Makes the h and J np arrays for use in creating the bqm and networkx graph
def make_h_J(C_i, C_ij, mu, sigma):
    """
    Creates the node (h) and edge (J) weightings for use in building the problem graph.

    Parameters:
    - C_i           Intermediate data representation needed for D-Wave's BQM format.
    - C_ij          Intermediate data representation needed for D-Wave's BQM format.
    - mu            Spins of qubits for this iteration.
    - sigma         zoom_factor to the power of the iteration.
    """
    h = 2 * sigma * (np.einsum("ij, j", C_ij, mu) - C_i)
    J = 2 * np.triu(C_ij, k=1) * pow(sigma, 2)

    return h, J


# Independent function for simplifying problem
# -> should hold a couple simple pruning methods
def default_prune(J, cutoff_percentile):
    """
    J is an adjacency matrix, but because our graph only has at most a single edge between two nodes,
    only the upper triangle of J encodes the edge weights (ie - J[2,3] is referring to the same edge as
    J[3,2] so we only store it once). As such, we need to compute the cutoff on only the upper triangle
    (the nonzero values) of J. Also, np.percentile doesn't take into account magnitude, only value, so
    in order to not lose all negative values, we use the absolute value of J.

    Parameters:
    - J                     Matrix of couplings between qubits
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed
    """
    shape = np.shape(J)
    flat_J = np.ndarray.flatten(J)
    abs_J = np.abs(flat_J)
    triu_abs_J = np.abs(J[np.triu_indices(shape[0])])
    cutoff_val = np.percentile(triu_abs_J, cutoff_percentile)
    new_J = np.where(abs_J > cutoff_val, flat_J, 0)

    return np.reshape(new_J, shape)


# makes a dwave bqm and corresponding networkx graph
def make_bqm(h, J, fix_vars, prune_vars, cutoff_percentile):
    """
    Creates the BQM and associated nx graph. The two differ in that fixed nodes
    are not removed from the nx graph.

    TODO: Clean up code and add in script tags for Graphs

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    - fix_vars              Defines whether to fix and remove low-variance qubits, as
                            described in D-Wave's fix_variables documentation.
    - prune_vars            Method used to remove weak couplings.
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed.
    """
    if prune_vars is not None:
        J = prune_vars(J, cutoff_percentile)

    bqm_nx = nx.from_numpy_array(J)

    GRAPHS = False
    if GRAPHS:
        run = 0
        while os.path.exists(
            "/Users/tsievert/QAMLZ Duarte/Misc/qamlz_for_higgs/Graphs/new_bef_graph_%03d.xml"
            % run
        ):
            run += 1
        nx.write_graphml(
            bqm_nx,
            "/Users/tsievert/QAMLZ Duarte/Misc/qamlz_for_higgs/Graphs/new_J_graph_%03d.xml"
            % run,
        )

    node_dict, attr_dict = {}, {}
    for node in bqm_nx:
        attr_dict["h_bias"] = h[node]
        node_dict[node] = attr_dict
    nx.set_node_attributes(bqm_nx, node_dict)

    if GRAPHS:
        nx.write_graphml(
            bqm_nx,
            "/Users/tsievert/QAMLZ Duarte/Misc/qamlz_for_higgs/Graphs/new_bef_graph_%03d.xml"
            % run,
        )

    bqm = dimod.from_networkx_graph(
        bqm_nx,
        vartype="SPIN",
        node_attribute_name="h_bias",
        edge_attribute_name="weight",
    )
    fixed_dict = None
    if fix_vars:
        lowerE, fixed_dict = dwplb.roof_duality(bqm, strict=True)
        if fixed_dict == {} or len(fixed_dict) < (np.size(h) // 2):
            print("lossely fixed")
            lowerE, fixed_dict = dwplb.roof_duality(bqm, strict=False)
        print("fixed_dict len = %d" % len(fixed_dict))
        for i in fixed_dict.keys():
            bqm.fix_variable(i, fixed_dict[i])
            bqm_nx.remove_node(i)
            bqm_nx.add_node(i, h_bias=fixed_dict[i])

    print(
        "nx node num = %d \ndw node num = %d"
        % (bqm_nx.number_of_nodes(), bqm.num_variables)
    )
    print(
        "nx edge num = %d \ndw edge num = %d"
        % (bqm_nx.number_of_edges(), bqm.num_interactions)
    )
    if GRAPHS:
        nx.write_graphml(
            bqm_nx,
            "/Users/tsievert/QAMLZ Duarte/Misc/qamlz_for_higgs/Graphs/new_aft_graph_%03d.xml"
            % run,
        )

    return bqm, bqm_nx, fixed_dict


# Independent function for quantum error correction
# -> should hold a single correction scheme to start (can always add)
def default_qac(h, J, fix_vars, prune_vars, cutoff_percentile, C, gamma):
    """
    Encodes the logical qubits with the NQAC method.

    Acts like a tiling for the J and con_J matrices -> places J along "main diagonal" of super-matrix
    (matrix of matrices) and con_J in off-diagonal of super-matrix

    TODO:

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    - fix_vars              Defines whether to fix and remove low-variance qubits, as
                            described in D-Wave's fix_variables documentation.
    - prune_vars            Method used to remove weak couplings.
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed.
    - C                     Same as encoding_depth: Defined for NQAC as the number of copies, can be made into
                            something else for other error-correction schemes.
    - gamma                 Defined for NQAC as the penalty associated with differing
                            spins for encoded-qubits corresponding to the same
                            logical qubit, can be made into something else
                            for other error-correction schemes.
    """
    rows, cols = np.shape(J)
    con_J = (
        J
        + J.T
        + (-gamma) * np.max(np.abs(np.ndarray.flatten(J))) * np.eye(rows, cols, k=0)
    )
    qa_J = np.zeros((C * rows, C * cols))
    for i in np.arange(C):
        for j in np.arange(i, C):
            if i == j:
                qa_J[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] = J
            else:
                qa_J[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] = con_J
    qa_h = np.repeat(h, C) * C

    return make_bqm(qa_h, qa_J, fix_vars, prune_vars, cutoff_percentile)


# Used to compare/benchmark the performance of the error correction
def default_copy(h, J, fix_vars, prune_vars, cutoff_percentile, C):
    """
    Encodes the logical qubits with the copy method.

    Parameters:
    - h                     Array of qubit-spin weights.
    - J                     Matrix of coupling strengths between qubits.
    - fix_vars              Defines whether to fix and remove low-variance qubits, as
                            described in D-Wave's fix_variables documentation.
    - prune_vars            Method used to remove weak couplings.
    - cutoff_percentile     Minimum percentile for couplings to NOT be removed.
    - C                     Same as encoding_depth: Defined for NQAC as the number of copies, can be made into
                            something else for other error-correction schemes.
    """
    rows, cols = np.shape(J)
    cp_J = np.zeros((C * rows, C * cols))
    for i in np.arange(C):
        cp_J[i * rows : (i + 1) * rows, i * cols : (i + 1) * cols] = J
    h = np.repeat(h, C)

    return make_bqm(h, cp_J, fix_vars, prune_vars, cutoff_percentile)


# adjust weights
def scale_weights(th, tJ, strength):
    """
    Scales the weightings dependent upon the iteration. It is unclear
    what purpose this serves.

    Parameters:
    - th            Embedded h array.
    - tJ            Embedded J matrix.
    - strength      Current-iteration-indexed strengths array.
    """
    for k in list(th.keys()):
        th[k] /= strength
    for k in list(tJ.keys()):
        tJ[k] /= strength

    return th, tJ


# Does the actual DWave annealing and connecting
def dwave_connect(config, iter, sampler, bqm, bqm_nx, anneal_time):
    """
    Handles all the connection to D-Wave, including:
    - Embedding the problem into a D-Wave architecture
    - Sending the problem to be annealed
    - Receiving the results from the annealing
    - Unembedding the problem from a D-Wave architecture
    - Unpacking the results into the obtained spins and their energies

    TODO: Multiply by a before and after embedding

    Parameters:
    - config            Config object that determines all training hyperparameters.
    - iter              Current iteration of the training.
    - sampler
    - bqm
    - bqm_nx
    - anneal_time
    """

    # a = np.sign(np.random.rand(nx.number_of_nodes(bqm_nx)) - 0.5)
    a = np.sign(np.random.rand(bqm.num_variables) - 0.5)
    FAST = False
    if FAST:
        if config.embedding is None:
            config.embedding = minorminer.find_embedding(
                bqm_nx, sampler.to_networkx_graph()
            )
    else:
        config.embedding = minorminer.find_embedding(
            bqm_nx, sampler.to_networkx_graph()
        )

    h_dict = nx.classes.function.get_node_attributes(bqm_nx, "h_bias")
    J_dict = nx.classes.function.get_edge_attributes(bqm_nx, "weight")
    # for k in h_dict:
    #     h_dict[k] *= a[k]
    # for u, v in J_dict:
    #     J_dict[(u, v)] *= a[u] * a[v]

    th, tJ = dwe.embed_ising(
        h_dict,
        J_dict,
        config.embedding,
        sampler.adjacency,
    )
    th, tJ = scale_weights(th, tJ, config.strengths[iter])

    qaresult = sampler.sample_ising(
        th,
        tJ,
        num_reads=config.nread,
        annealing_time=anneal_time,
        answer_mode="histogram",
    )
    unembed_qaresult = dw.embedding.unembed_sampleset(qaresult, config.embedding, bqm)

    a = np.tile(a, (np.size(unembed_qaresult.record.sample, axis=0), 1))
    samples, energies = (
        unembed_qaresult.record.sample * a,
        unembed_qaresult.record.energy,
    )

    return samples, energies


# Modifies results by undoing variable fixing
def unfix(spin_samples, h_len, fixed_dict):
    """
    Unpacks the qubits fixed using fix_vars.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - h_len                 Number of nodes in the graph sent to D-Wave.
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    spin_samples = np.zeros((np.size(spin_samples, 0), h_len))
    j = 0
    for i in range(h_len):
        if i in fixed_dict:
            spin_samples[:, i] = fixed_dict[i]
        else:
            spin_samples[:, i] = spin_samples[:, j]
            j += 1

    return spin_samples


# Modifies results by undoing variable encoding (qac method)
#   -> uses majority voting
def decode_qac(spin_samples, enc_h_len, orig_len, fix_vars, fixed_dict=None):
    """
    Decodes the qubits encoded with the NQAC method.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - enc_h_len             Number of nodes in the graph sent to D-Wave (after NQAC encoding).
    - orig_len              Number of nodes originally in the graph (before NQAC encoding).
    - fix_vars              Boolean of whether to fix any nodes using D-Wave's method.
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    if fix_vars:
        spin_samples = unfix(spin_samples, enc_h_len, fixed_dict=fixed_dict)
        spin_samples = np.where(spin_samples == 3, 0, spin_samples)
    true_spins = np.zeros((np.size(spin_samples, 0), orig_len))
    for i in range(orig_len):
        qubit_copies = []
        for j in range(np.size(spin_samples, axis=1) // orig_len):
            qubit_copies.append(spin_samples[:, j * orig_len])
        true_spins[:, i] = np.sign(np.sum(np.array(qubit_copies), axis=0))
    true_spins = np.where(true_spins == 0, np.random.choice([-1, 1]), true_spins)

    return true_spins


# Modifies results by undoing variable encoding (copy method)
def decode_copy(spin_samples, h_len, orig_len, fix_vars, fixed_dict=None):
    """
    Decodes the qubits encoded with the copy method.

    Parameters:
    - spin_samples          Obtained spins of all reads from D-Wave.
    - enc_h_len             Number of nodes in the graph sent to D-Wave (after copying).
    - orig_len              Number of nodes originally in the graph (before NQAC encoding).
    - fix_vars              Boolean of whether to fix any nodes using D-Wave's method.
    - fixed_dict            Dictionary of fixed nodes (fixed just prior to sending
                            D-Wave the graph in the dwave_connect function).
    """
    if fix_vars:
        spin_samples = unfix(spin_samples, h_len, fixed_dict=fixed_dict)

    multi_spins = np.zeros(
        (np.size(spin_samples, axis=1) // orig_len)
    )  # will be an exact int, not a floating point
    for i in range(np.size(multi_spins)):
        multi_spins[i] = spin_samples[:, i * orig_len : (i + 1) * orig_len]

    return multi_spins


def anneal(config, iter, env, mu):
    """Controls the annealing and energy selection process.

    TODO: split into multiple (two?) functions, depending on regular mode or qac mode

    Called from train() multiple times. Each call to anneal performs a training step
    on the D-Wave hardware.

    Parameters:
    - config       Configuration struct (hold many useful config params)
    - iter         Declares the current iteration within the training method. This is needed
                   as some configuration parameters are per-iteration arrays.
    - env          Env object that determines all data-processing hyperparameters.
    - mu           Vector of expected values of qubit spins.
    """
    h, J = make_h_J(env.C_i, env.C_ij, mu, pow(config.zoom_factor, iter))
    if config.encode_vars is None:
        bqm, bqm_nx, fixed_dict = make_bqm(
            h, J, config.fix_vars, config.prune_vars, config.cutoff_percentile
        )

        # if len(fixed_dict) != np.size(h):
        #     spin_samples, energies = dwave_connect(
        #         config, iter, env.sampler, bqm, bqm_nx, config.anneal_time
        #     )
        # else:
        #     spin_samples, energies = [], 0
        spin_samples, energies = dwave_connect(
            config, iter, env.sampler, bqm, bqm_nx, config.anneal_time
        )
        print(
            "min energy = %d, max energy = %d" % (np.amin(energies), np.amax(energies))
        )

        if config.fix_vars is not None:
            spin_samples = unfix(spin_samples, np.size(h), fixed_dict=fixed_dict)
        if np.size(spin_samples, axis=0) > config.max_states[iter]:
            excited_states_arr = [spin_samples[-config.max_states[iter] :]]
        else:
            excited_states_arr = [spin_samples]
    elif config.encode_vars is not None:
        orig_len = np.size(h)
        bqm, bqm_nx, fixed_dict = config.encode_vars(
            h,
            J,
            config.fix_vars,
            config.prune_vars,
            config.cutoff_percentile,
            config.encoding_depth,
            config.gamma,
        )
        # if len(fixed_dict) != config.encoding_depth * orig_len:
        #     spin_samples, energies = dwave_connect(
        #         config, iter, env.sampler, bqm, bqm_nx, config.anneal_time
        #     )
        # else:
        #     spin_samples, energies = [], 0
        spin_samples, energies = dwave_connect(
            config, iter, env.sampler, bqm, bqm_nx, config.anneal_time
        )
        spin_samples = config.decode_vars(
            spin_samples,
            config.encoding_depth * orig_len,
            orig_len,
            config.fix_vars,
            fixed_dict=fixed_dict,
        )
        if len(np.shape(spin_samples)) == 2:
            print(
                "min energy = %d, max energy = %d"
                % (np.amin(energies), np.amax(energies))
            )
            if np.size(spin_samples, axis=0) > config.max_states[iter]:
                excited_states_arr = [spin_samples[-config.max_states[iter] :]]
            else:
                excited_states_arr = [spin_samples]
        elif len(np.shape(spin_samples)) == 3:
            excited_states_arr = []
            if np.size(spin_samples, axis=1) > config.max_states[iter]:
                for i in range(np.size(spin_samples, 0)):
                    print(
                        "%d: min energy = %d, max energy = %d"
                        % (i, np.amin(energies[i]), np.amax(energies[i]))
                    )
                    excited_states_arr[i] = spin_samples[i][-config.max_states[iter] :]
            else:
                for i in range(np.size(spin_samples, 0)):
                    excited_states_arr[i] = spin_samples[i]

    return excited_states_arr
