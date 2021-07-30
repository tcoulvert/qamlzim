#!/usr/bin/env python3

import datetime
import os
import sys
import time

import numpy as np
from scipy.optimize import basinhopping
from sklearn.metrics import accuracy_score
#sys.exit(1)

from contextlib import closing

USE_DWAVE = False

if USE_DWAVE:
    # potentially don't even need to convert btwn qubo and ising. new var fixing stored under general class of BQM?
    # from dwave_sapi2.util import qubo_to_ising, ising_to_qubo
    from dimod import fix_variables, BinaryQuadraticModel as BQM
    import dimod
    # from dwave_sapi2.fix_variables import fix_variables
    from multiprocessing import Pool
    from dwave.cloud import Client 
    # import dwave_sapi2.remote
    from minorminer import find_embedding
    from dwave.embedding import embed_ising, unembed_sampleset
    # from dwave_sapi2.embedding import find_embedding,embed_problem,unembed_answer
    from dwave.system.samplers import DWaveSampler
    # from dwave_sapi2.util import get_hardware_adjacency
    # need to find where to put in new ocean tools for solve_ising, requires same(?) params so should be simple
    #from dwave_sapi2.core import solve_ising, await_completion


a_time = 5
train_sizes = [100, 1000, 5000, 10000, 15000, 20000]
start_num = 9
end_num = 10

zoom_factor = 0.5
n_iterations = 1 
# n_iterations = 8

flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))/2
flip_state = -1

AUGMENT_CUTOFF_PERCENTILE = 95
AUGMENT_SIZE = 7   # must be an odd number (since augmentation includes original value in middle)
AUGMENT_OFFSET = 0.0075

FIXING_VARIABLES = True

# Entire function is broken, none of the vars/methods are defined, they are only referenced, potentially don't need???
def hamiltonian_checker(s, C_i, C_ij, reg):
    if POS_WEIGHTS:
        return hamiltonian_orig_posweights(s, C_i, C_ij, reg)
    elif POS_NEG_WEIGHTS:
        return hamiltonian_orig_posnegweights(s, C_i, C_ij, reg)

def total_hamiltonian(s, C_i, C_ij):
    bits = len(s)
    h = 0 - np.dot(s, C_i)
    for i in range(bits):
        h += s[i] * np.dot(s[i+1:], C_ij[i][i+1:])
    return h
    
def hamiltonian(s, C_i, C_ij, mu, sigma, reg):
    s[np.where(s > 1)] = 1.0
    s[np.where(s < -1)] = -1.0
    bits = len(s)
    h = 0
    for i in range(bits):
        h += 2*s[i]*(-sigma[i]*C_i[i])
        for j in range(bits):
            if j > i:
                h += 2*s[i]*s[j]*sigma[i]*sigma[j]*C_ij[i][j]
            h += 2*s[i]*sigma[i]*C_ij[i][j] * mu[j]
    return h


def anneal(C_i, C_ij, mu, sigma, l, strength_scale, energy_fraction, ngauges, max_excited_states):
    url = "https://cloud.dwavesys.com/sapi"
    token = os.environ["DWAVE_TOKEN"]
    if not len(token):
        print("error getting token")
    
    h = np.zeros(len(C_i))
    J = {}
    for i in range(len(C_i)):
        h_i = -2*sigma[i]*C_i[i]
        for j in range(len(C_ij[0])):
            if j > i:
                J[(i, j)] = 2*C_ij[i][j]*sigma[i]*sigma[j]
            h_i += 2*(sigma[i]*C_ij[i][j]*mu[j]) 
        h[i] = h_i

    vals = np.array(J.values())
    cutoff = np.percentile(vals, AUGMENT_CUTOFF_PERCENTILE)
    to_delete = []
    for k, v in J.items():
        if v < cutoff:
            to_delete.append(k)
    for k in to_delete:
        del J[k]

    isingpartial = []

    # Likely need to completely re-do
    if FIXING_VARIABLES:
        bqm = BQM.from_ising(h, J, offset) #potentially need to rework in order for func to get params
        # "offset" not declared, only referenced, bc might 
        #  be desired in new embed func but wasn't possible in previous func
        fixed_dict = dimod.fix_variables(bqm)
        # end of new Ocean SDK
        Q, _  = ising_to_qubo(h, J)
        simple = fix_variables(Q, method='standard')
        new_Q = simple['new_Q']
        print('new length', len(new_Q))
        isingpartial = simple['fixed_variables']
    if (not FIXING_VARIABLES) or len(new_Q) > 0:
        cant_connect = True
        while cant_connect:
            try:
                print('about to call remote')
                conn = Client(endpoint=url, token=token)
                sampler = DWaveSampler(endpoint = url, token = token) 
                # solver = conn.get_solver("DW2X")
                print('called remote', conn)
                cant_connect = False
            except IOError:
                print('Network error, trying again', datetime.datetime.now())
                time.sleep(10)
                cant_connect = True

        A = sampler.adjacency 
        # A = get_hardware_adjacency(solver)
        
        mapping = []
        offset = 0
        for i in range(len(C_i)):
            if i in isingpartial:
                mapping.append(None)
                offset += 1
            else:
                mapping.append(i - offset)
        if FIXING_VARIABLES:
            new_Q_mapped = {}
            for (first, second), val in new_Q.items():
                new_Q_mapped[(mapping[first], mapping[second])] = val
            h, J, _ = qubo_to_ising(new_Q_mapped)
        
        # run gauges
        nreads = 200
        qaresults = np.zeros((ngauges*nreads, len(h)))
        for g in range(ngauges):
            embedded = False
            for attempt in range(5):
                a = np.sign(np.random.rand(len(h)) - 0.5)
                h_gauge = h*a
                J_gauge = {}
                for i in range(len(h)):
                    for j in range(len(h)):
                        if (i, j) in J:
                            J_gauge[(i, j)] = J[(i, j)]*a[i]*a[j]
            
                embedding = find_embedding(J.keys(), A)
                try:
                    # Likely need to completely re-do
                    th, tJ = embed_ising(h_gauge, J_gauge, embedding, target) 
                    # "target" not declared, only referenced, bc might 
                    #  be desired in new embed func but wasn't possible in previous func
                    # (h0, j0, jc, new_emb) = embed_problem(h_gauge, J_gauge, embeddings, A, True, True)
                    embedded = True
                    break
                except ValueError:      # no embedding found
                    print('no embedding found')
                    embedded = False
                    continue
            
            if not embedded:
                continue
            
            # adjust chain strength
            for k, v in j0.items(): # "j0" not returned anymore bc embed func changed
                j0[k] /= strength_scale # "j0" not returned anymore bc embed func changed
            for i in range(len(h0)): # "h0" not returned anymore bc embed func changed
                h0[i] /= strength_scale # "h0" not returned anymore bc embed func changed

            emb_j = j0.copy() # "j0" not returned anymore bc embed func changed
            emb_j.update(jc) # "jc" not returned anymore bc embed func changed
        
            print("Quantum annealing")
            try_again = True
            while try_again:
                try:
                    sampler = DWaveSampler()
                    response = sampler.sample_ising(h0, emb_j, num_reads = nreads, annealing_time = a_time, answer_mode='raw')
                    # "h0" not returned anymore bc embed func changed
                    # qaresult = solve_ising(solver, h0, emb_j, num_reads = nreads, annealing_time = a_time, answer_mode='raw')
                    try_again = False
                except:
                    print('runtime or ioerror, trying again')
                    time.sleep(10)
                    try_again = True
            print("Quantum done")

            samples = unembed_sampleset(embedded, embedding, bqm)
            # qaresult = np.array(unembed_answer(qaresult["solutions"], new_emb, 'vote', h_gauge, J_gauge))
            qaresult = qaresult * a
            qaresults[g*nreads:(g+1)*nreads] = qaresult
        
        if FIXING_VARIABLES:
            j = 0
            for i in range(len(C_i)):
                if i in isingpartial:
                    full_strings[:, i] = 2*isingpartial[i] - 1 # "full_strings" not declared, only referenced
                else:
                    full_strings[:, i] = qaresults[:, j] # "full_strings" not declared, only referenced
                    j += 1
        else:
            full_strings = qaresults # "full_strings" not declared, only referenced
        
        s = full_strings # "full_strings" not declared, only referenced
        energies = np.zeros(len(qaresults))
        s[np.where(s > 1)] = 1.0
        s[np.where(s < -1)] = -1.0
        bits = len(s[0])
        for i in range(bits):
            energies += 2*s[:, i]*(-sigma[i]*C_i[i])
            for j in range(bits):
                if j > i:
                    energies += 2*s[:, i]*s[:, j]*sigma[i]*sigma[j]*C_ij[i][j]
                energies += 2*s[:, i]*sigma[i]*C_ij[i][j] * mu[j]
        
        unique_energies, unique_indices = np.unique(energies, return_index=True)
        ground_energy = np.amin(unique_energies)
        if ground_energy < 0:
            threshold_energy = (1 - energy_fraction) * ground_energy
        else:
            threshold_energy = (1 + energy_fraction) * ground_energy
        lowest = np.where(unique_energies < threshold_energy)
        unique_indices = unique_indices[lowest]
        if len(unique_indices) > max_excited_states:
            sorted_indices = np.argsort(energies[unique_indices])[-max_excited_states:]
            unique_indices = unique_indices[sorted_indices]
        final_answers = full_strings[unique_indices] # "full_strings" not declared, only referenced
        print('number of selected excited states', len(final_answers))
        
        return final_answers
        
    else:
        final_answer = []
        for i in range(len(C_i)):
            if i in isingpartial:
                final_answer.append(2*isingpartial[i] - 1)
        final_answer = np.array(final_answer)
        return np.array([final_answer])

# Currently not used, augmented_data used instead
def create_data(sig, bkg):
    n_classifiers = sig.shape[1]
    predictions = np.concatenate((np.sign(sig), np.sign(bkg)))
    predictions = np.transpose(predictions) / float(n_classifiers)
    y = np.concatenate((np.ones(len(sig)), -np.ones(len(bkg))))
    return predictions, y

def create_augmented_data(sig, bkg): # sig and bkg are only the portions sampled this iteration, out of the total sig and bkg
    offset = AUGMENT_OFFSET
    scale = AUGMENT_SIZE

    n_samples = len(sig) + len(bkg) # total number of sampled data points
    n_classifiers = sig.shape[1] # ???
    predictions_raw = np.concatenate((sig, bkg)) # combine the two arrays into a larger array
    predictions_raw = np.transpose(predictions_raw) # ???
    predictions = np.zeros((n_classifiers * scale, n_samples)) # make an array of zeroes at some length???
    for i in range(n_classifiers):
        for j in range(scale):
            predictions[i*scale + j] = np.sign(predictions_raw[i] + (j-scale//2)*offset) / (n_classifiers * scale)
    y = np.concatenate((np.ones(len(sig)), -np.ones(len(bkg))))
    print('predictions', predictions)
    return predictions, y

# Does something like combining the training into one place???
def ensemble(predictions, weights):
    ensemble_predictions = np.zeros(len(predictions[0]))
    
    if POS_NEG_WEIGHTS: # "POS_NEG_WEIGHTS" not declared, only referenced
        return np.sign(np.dot(predictions.T, weights))
    else:
        return np.sign(np.dot(predictions.T, weights/2 + 0.5)/n_classifiers)

# Step 1: Load Background and Signal data
print('loading data')
sig = np.loadtxt('sig.csv')
bkg = np.loadtxt('bkg.csv')
# 1.1 - After loading, Calculate the percentage of bkg and sig out of the total data (combination of the two)
sig_pct = float(len(sig)) / (len(sig) + len(bkg))
bkg_pct = float(len(bkg)) / (len(sig) + len(bkg))
print('loaded data')

# Step 2: Initialize n-folds variable and num outside of the for-loop
n_folds = 10
num = 0

#Step 3: Loop over all the different training sizes (train_size declared/defined below imports)
for train_size in train_sizes:
    print('training with size', train_size)
    # 3.1 - Create arrays with sizes equal to the sizes of bkg and sig
    sig_indices = np.arange(len(sig))
    bkg_indices = np.arange(len(bkg))
    
    # 3.2 - Create indexing vars with sizes equal to the sizes of the recently created "indices" arrays, and bkg and sig datas
    remaining_sig = sig_indices
    remaining_bkg = bkg_indices

    # 3.3 - generate folds(what are folds??) using 0 as the seed -> potentially use a different quantum algorithm
    #        to generate actual random numbers
    fold_generator = np.random.RandomState(0)

    #Step 5: iterate over the number of folds
    for f in range(n_folds):
        if num >= end_num: # end_num declared/defined below imports 
            break
        print('fold', f)
        # 5.1 - returns "train" arrays by randomly sampling some of "reamining" arrays -> the amount of some 
        #        determined by the iteration of train_size * data pct. "train" arrays store the indeces that were 
        #        randomly selected to train with that iteration
        train_sig = fold_generator.choice(remaining_sig, size=int(train_size*sig_pct), replace=False)
        train_bkg = fold_generator.choice(remaining_bkg, size=int(train_size*bkg_pct), replace=False)
        
        # 5.2 - update "remaining" arrays for next loop by deleting the indeces of the data sampled that iteration, 
        #        so this stores a long-term record of unused indeces (a record across all iterations)
        remaining_sig = np.delete(remaining_sig, train_sig)
        remaining_bkg = np.delete(remaining_bkg, train_bkg)
        
        # 5.3 - create/overwrite "test" arrays by temporarily deleting the indeces of the data sampled that iteration, 
        #        however at the next iteration test_sig will be overwritten to only have deleted the indeces used that 
        #        iteration, so this stores a short-term record of unused indeces (a record only from the current iteration)
        test_sig = np.delete(sig_indices, train_sig)
        test_bkg = np.delete(bkg_indices, train_bkg)

        # 5.4 - creates "train" and "test" vars by sending the pieces of the data arrays ("sig" and "bkg") as 
        #        determined by the indeces produced from the random samplings ("train" and "test" arrays), 
        #        so as time goes on the "train" arrays will get smaller, but the "test" arrays will stay the same size
        predictions_train, y_train = create_augmented_data(sig[train_sig], bkg[train_bkg])
        predictions_test, y_test = create_augmented_data(sig[test_sig], bkg[test_bkg])
        print('split data')
        
        # 5.5 - increment num
        if num < start_num:
            num += 1
            continue

        # 5.6 - create C_ij and C_i matrices
        n_classifiers = len(predictions_train)
        C_ij = np.zeros((n_classifiers, n_classifiers))
        C_i = np.dot(predictions_train, y_train)
        for i in range(n_classifiers):
            for j in range(n_classifiers):
                C_ij[i][j] = np.dot(predictions_train[i], predictions_train[j])

        print('created C_ij and C_i matrices')

        # 5.7 - Create/update ML vars based on matrices
        mu0 = np.zeros(n_classifiers)
        sigma0 = np.ones(n_classifiers)
        mu = np.copy(mu0)
        sigma = np.copy(sigma0)
        reg = 0.0
        l0 = reg*np.amax(np.diagonal(C_ij)*sigma*sigma - 2*sigma*C_i)
        strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        gauges = [50, 10] + [1]*(n_iterations - 2)
        max_states = [16, 4] + [1]*(n_iterations - 2)     # cap the number of excited states accepted per iteration

        mus = [np.zeros(n_classifiers)]
        iterations = n_iterations
        for i in range(iterations):
            print('iteration', i)
            l = reg*np.amax(np.diagonal(C_ij)*sigma*sigma - 2*sigma*C_i)
            new_mus = []
            for mu in mus:
                excited_states = anneal(C_i, C_ij, mu, sigma, l, strengths[i], energy_fractions[i], gauges[i], max_states[i])
                for s in excited_states:
                    new_energy = total_hamiltonian(mu + s*sigma*zoom_factor, C_i, C_ij) / (train_size - 1)
                    flips = np.ones(len(s))
                    for a in range(len(s)):
                        temp_s = np.copy(s)
                        temp_s[a] = 0
                        old_energy = total_hamiltonian(mu + temp_s*sigma*zoom_factor, C_i, C_ij) / (train_size - 1)
                        energy_diff = new_energy - old_energy
                        if energy_diff > 0:
                            flip_prob = flip_probs[i]
                            flip = np.random.choice([1, flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[a] = flip
                        else:
                            flip_prob = flip_others_probs[i]
                            flip = np.random.choice([1, flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[a] = flip
                    flipped_s = s * flips
                    new_mus.append(mu + flipped_s*sigma*zoom_factor)
            sigma *= zoom_factor
            mus = new_mus
            
            np.save('./mus' + str(try_number) + '/mus' + str(train_size) + 'fold' + str(f) + 'iter' + str(i) + '.npy', np.array(mus))
            # "try_number" not declared, only referenced
        for mu in mus:
            print('final accuracy on train set', accuracy_score(y_train, ensemble(predictions_train, mu)))
            print('final accuracy on test set', accuracy_score(y_test, ensemble(predictions_test, mu)))
        num += 1
