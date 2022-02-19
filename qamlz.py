#!/usr/bin/env python3
import datetime
import json
import logging
import os
import subprocess
import sys
import time

import abc
import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np
import sklearn as sk

from copy import deepcopy
from dwave.system.samplers import DWaveSampler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedShuffleSplit

# Numpy arrays should be row-major for best performance

def total_hamiltonian(s, C_i, C_ij):
    bits = len(s)
    h = 0 - np.dot(s, C_i)
    for i in range(bits):
        h += s[i] * np.dot(s[i+1:], C_ij[i][i+1:])
    return h

def make_h_J(C_i, C_ij, mu, sigma):
    h = 2 * sigma * (np.einsum('ij, jk', C_ij, mu) - C_i)
    J = 2 * np.triu(C_ij, k=1) * pow(sigma, 2)
    
    return h, J

# Independent function for simplifying problem
# -> should hold a couple simple pruning methods
def default_prune(J, cutoff_percentile):
    cutoff_plus = np.percentile(J, cutoff_percentile)
    cutoff_minus = np.percentile(-J, cutoff_percentile)

    return J[np.logical_or(J > cutoff_plus, J < cutoff_minus)]

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
def scale_weights(th, tJ, strength_scale):
    for k in list(th.keys()):
        th[k] /= strength_scale 
    for k in list(tJ.keys()):
        tJ[k] /= strength_scale

    return th, tJ

def anneal(C_i, C_ij, mu, sigma, strength_scale, energy_fraction, ngauges, max_excited_states, A_adj, A, sampler, prune_vars, cutoff, encode_vars, fix_vars, nreads, anneal_time=5):
    h, J = make_h_J(C_i, C_ij, mu, sigma)
    if not prune_vars is None:
        J = prune_vars(J, cutoff)
    if encode_vars is None:
        bqm, bqm_networkx, fixed_dict = make_bqm(h, J, fix_vars)
    else:
        orig_len = np.size(h)
        bqm, bqm_networkx, fixed_dict = encode_vars(h, J, fix_vars)

    num_nodes = bqm.num_variables
    qaresults = np.zeros((ngauges*nreads, num_nodes))
    for g in range(ngauges):
        a = np.sign(np.random.rand(num_nodes) - 0.5)
        embedding = minorminer.find_embedding(bqm_networkx, A)
        th, tJ = dw.embedding.embed_ising(nx.classes.function.get_node_attributes(bqm_networkx, 'h_bias'), nx.classes.function.get_edge_attributes(bqm_networkx, 'J_bias'), embedding, A_adj)
        th, tJ = scale_weights(th, tJ, strength_scale)

        qaresult = sampler.sample_ising(th, tJ, num_reads = nreads, annealing_time=anneal_time, answer_mode='raw')
        unembed_qaresult = dw.embedding.unembed_sampleset(qaresult, embedding, bqm)

        for i in range(len(unembed_qaresult.record.sample)):
            unembed_qaresult.record.sample[i, :] = unembed_qaresult.record.sample[i, :] * a
        qaresults[g*nreads:(g+1)*nreads] = unembed_qaresult.record.sample

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

    s = full_strings 
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
    
    unique_energies, unique_indices = np.unique(en_energies, return_index=True)
    ground_energy = np.amin(unique_energies)
    if ground_energy < 0:
        threshold_energy = (1 - energy_fraction) * ground_energy
    else:
        threshold_energy = (1 + energy_fraction) * ground_energy
    lowest = np.where(unique_energies < threshold_energy)
    unique_indices = unique_indices[lowest]
    if len(unique_indices) > max_excited_states:
        sorted_indices = np.argsort(en_energies[unique_indices])[-max_excited_states:]
        unique_indices = unique_indices[sorted_indices]
    final_answers = full_strings[unique_indices]
    print('number of selected excited states', len(final_answers))
    
    return final_answers

class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    # X_train, y_train should be formatted like scikit-learn data arrays are
    # -> X_train is the train data, y_train is the train labels
    def __init__(self, X_train, y_train, X_validation=None, y_validation=None, fidelity=7):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        
        self.fidelity = fidelity
        self.fidelity_offset = 0.0225 / fidelity

        self.C_i = None
        self.C_ij = None
        self.data_preprocess()

    # Splits training data into train and validation 
    # -> validation allows hyperparameter adjustment
    def sklearn_data_wrapper(self, sk_model):
        X_tra, X_val = [], []
        y_tra, y_val = [], []
        for train_indices, validation_indices in sk_model.split(self.X_train, self.y_train):
            X_tra.append(self.X_train[train_indices]), X_val.append(self.X_train[validation_indices])
            y_tra.append(self.y_train[train_indices]), y_val.append(self.y_train[validation_indices])
        
        self.X_train, self.X_validation = np.array(X_tra), np.array(X_val)
        self.y_train, self.y_validation = np.array(y_tra), np.array(y_val)

    """
        This duplicates the parameters 'fidelity' times. The purpose is to turn the weak classifiers
        from outputing a single number (-1 or 1) to outputting a binary array ([-1, 1, 1,...]). The 
        use of such a change is to trick the math into allowing more nuance between a weak classifier
        that outputs 0.1 from a weak classifier that outputs 0.9 (the weak classifier outputs are continuous)
        -> thereby discretizing the weak classifier's decision into more pieces than binary.
    """
    """
        This creates a periodic array to shift the outputs of the repeated weak classifier, so that there
        is a meaning to duplicating them. You can think of each successive digit of the resulting weak classifier
        output array as being more specific about what the continuous output was - ie >0, >0.1, >0.2 etc. This 
        description is not exactly correct in this case but it is the same idea as what we're doing.
    """
    def data_preprocess(self):
        m_events, n_params = np.shape(self.X_train) # [M events (rows) x N parameters (columns)]

        c_i = np.repeat(self.X_train, repeats=self.fidelity, axis=1) # [M events (rows) x N*fidelity parameters (columns)]
        
        offset_array = self.fidelity_offset * (np.tile(np.arange(self.fidelity), m_events * n_params) - self.fidelity//2)
        c_i = np.sign(np.ndarray.flatten(c_i, order='C') - offset_array) / (n_params * self.fidelity)
        c_i = np.reshape(c_i, (m_events, n_params*self.fidelity))

        C_i = np.dot(self.y_train, c_i)
        C_ij = np.einsum('ij,kj', c_i, c_i)

        self.C_i, self.C_ij = C_i, C_ij

class ModelConfig:
    def __init__(self, n_iterations=10, zoom_factor=0.5):
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor
        self.fix_var = True
        
        self.flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        self.flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4)) / 2
        FLIP_STATE = -1

        self.strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        self.energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        self.gauges = [50, 10] + [1]*(n_iterations - 2)
        self.nreads = 200
        self.max_states = [16, 4] + [1]*(n_iterations - 2)

        self.fix_vars = True
        self.prune_vars = default_prune
        self.default_cutoff = 95
        self.encode_vars = default_qac

class Model:
    def __init__(self, config, endpoint_url, account_token):
        # add in hyperparameters in ModelConfig
        # -> this is where user determines how the model will train
        self.config = config
        self.sampler = None
        self.create_sampler(endpoint_url, account_token)
        
        self.anneal_results = {}
        self.mus_dict = {}
        
    def create_sampler(self, endpoint_url, account_token, name='Advantage_system4.2'):
        while cant_connect:
            try:
                self.sampler = DWaveSampler(endpoint=endpoint_url, token=account_token, solver=name)
                cant_connect = False
            except IOError:
                time.sleep(10)
                cant_connect = True

    def train(self, env):
        C_i, C_ij = env.data_preprocess(self.config.fidelity_offset, self.config.fidelity)
        A_adj, A = self.sampler.adjacency, self.sampler.to_networkx_graph()
        mus = [np.zeros(np.size(C_i))]
        train_size = np.shape(env.X_train)[0]

        for i in range(self.config.n_iterations):
            sigma = pow(self.config.zoom_factor, i)
            new_mus = []
            for mu in mus:
                excited_states = anneal(C_i, C_ij, mu, sigma, self.config.strengths[i], self.config.energy_fractions[i], self.config.gauges[i], self.config.max_states[i], A_adj, A, self.sampler, self.config.prune_vars, self.config.default_cutoff, self.config.encode_vars, self.config.fix_var, self.config.nreads)
                for s in excited_states:
                    new_energy = total_hamiltonian(mu + s*sigma*self.config.zoom_factor, C_i, C_ij) / (train_size - 1)
                    flips = np.ones(len(s))
                    temp_s = np.copy(s)
                    for a in range(len(s)):
                        temp_s[a] = 0
                        old_energy = total_hamiltonian(mu + temp_s*sigma*self.config.zoom_factor, C_i, C_ij) / (train_size - 1)
                        energy_diff = new_energy - old_energy
                        if energy_diff > 0:
                            flip_prob = self.config.flip_probs[i]
                            flip = np.random.choice([1, self.config.flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[a] = flip
                        else:
                            flip_prob = self.config.flip_others_probs[i]
                            flip = np.random.choice([1, self.config.flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[a] = flip
                    flipped_s = s * flips
                    new_mus.append(mu + flipped_s*sigma*self.config.zoom_factor)
                sigma *= self.config.zoom_factor
                mus = new_mus
                
                mus_filename = 'mus%05d_iter%d-%s.npy' % (train_size, i, datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S'))
                # mus_destdir = os.path.join(script_path, 'mus')
                # mus_filepath = (os.path.join(mus_destdir, mus_filename))
                # if not os.path.exists(mus_destdir):
                #     os.makedirs(mus_destdir)
                # np.save(mus_filepath, np.array(mus))
                self.mus_dict[mus_filename].append(np.array(mus))
            avg_arr_train =[]
            for mu in mus:
                avg_arr_train.append(accuracy_score(env.y_train, self.evaluate(env.X_train, mu)))
            self.anneal_results[mus_filename].append(np.mean(np.array(avg_arr_train)))
            num += 1

    def evaluate(self, X_data, weights):
        # split data up and run model on test data
        # return avg accuracy and std dev (from subsets)
        ensemble_predictions = np.zeros(len(X_data[0]))
    
        return np.sign(np.dot(X_data.T, weights))


# For now, keep outside package
# -> formats data from higgs-specific CSVs, not general
def format_data(sig, bkg, sk_model, n_splits=10):
    sig = np.array(np.split(sig, n_splits))
    bkg = np.array(np.split(bkg, n_splits))
    X_temp = np.concatenate(sig, bkg)
    y_temp = np.concatenate(np.ones(sig.size), -np.ones(bkg.size))

    X_tra, X_val = [], []
    y_tra, y_val = [], []
    for train_indices, validation_indices in sk_model.split(X_temp, y_temp):
        X_tra.append(X_temp[train_indices]), X_val.append(X_temp[validation_indices])
        y_tra.append(y_temp[train_indices]), y_val.append(y_temp[validation_indices])

    return (np.array(X_tra), np.array(X_val), np.array(y_tra), np.array(y_val))


X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])

X = np.array([[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17]])
y = np.array([0, 0, 0, 1, 1, 1])

X = np.arange(20)
X = np.array(np.split(X, 5))
print(X)
y = np.array([0, 0, 1, 1, 1])

env = TrainEnv(X, y)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
env.sklearn_data_wrapper(sss)

bin_config = ModelConfig()
bin_class = Model()
bin_class.train(env)