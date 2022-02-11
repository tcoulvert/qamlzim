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
import networkx as nx
import numpy as np
import sklearn as sk

from dwave.system.samplers import DWaveSampler
from sklearn.model_selection import StratifiedShuffleSplit

# Numpy arrays should be row-major for best performance

class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    # X_train, y_train should be formatted like scikit-learn data arrays are
    # -> X_train is the train data, y_train is the train labels
    def __init__(self, X_train, y_train, X_validation=None, y_validation=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation

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

class Model:
    def __init__(self, config):
        # add in hyperparameters in ModelConfig
        # -> this is where user determines how the model will train
        self.config = config
        

    def evaluate(self, test_sig, test_bkg):
        # split data up and run model on test data
        # return avg accuracy and std dev (from subsets)
        
        return

class ModelConfig:
    def __init__(self, n_iterations=10, zoom_factor=0.5, augment_size=7):
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor
        self.augment_size = augment_size
        self.augment_offset = 0.0225 / augment_size

        self.flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        self.flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4)) / 2
        FLIP_STATE = -1

        self.strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        self.energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        self.gauges = [50, 10] + [1]*(n_iterations - 2)
        self.max_states = [16, 4] + [1]*(n_iterations - 2)
        

# Allows for basis of many hardware classes 
# -> can be easily programmed to run on systems besides DWave
class Hardware(abc.ABC):
    @abc.abstractmethod
    def convert_from_networkx(self):
        pass

    @abc.abstractmethod
    def data_preprocessing(self):
        pass

# Specific class for DWave hardware
class DWaveQA(Hardware,dimod.BinaryQuadraticModel):
    def __init__(self, name='Advantage_system4.2'):
        self.name = name
        self.sampler = None

    # Hardware-depdendent function the pre- and post-processing on the input data
    # -> should do the folding and iterating
    # -> only needed for QA, not necessary in general
    def data_processing(self, model_config, X_data):
        offset = model_config.augment_offset
        fidelity = model_config.augment_size
        m_events, n_params = np.shape(X_data) # [M events (rows) x N parameters (columns)]
        """
            This duplicates the parameters 'fidelity' times. The purpose is to turn the weak classifiers
            from outputing a single number (-1 or 1) to outputting a binary array ([-1, 1, 1,...]). The 
            use of such a change is to trick the math into allowing more nuance between a weak classifier
            that outputs 0.1 from a weak classifier that outputs 0.9 (the weak classifier outputs are continuous)
            -> thereby discretizing the weak classifier's decision into more pieces than binary.
        """
        c_i = np.repeat(X_data, repeats=fidelity, axis=1) # [M events (rows) x N*fidelity parameters (columns)]
        """
            This creates a periodic array to shift the outputs of the repeated weak classifier, so that there
            is a meaning to duplicating them. You can think of each successive digit of the resulting weak classifier
            output array as being more specific about what the continuous output was - ie >0, >0.1, >0.2 etc. This 
            description is not exactly correct in this case but it is the same idea as what we're doing.
        """
        offset_array = offset * (np.tile(np.arange(fidelity), m_events * n_params) - fidelity//2)
        c_i = np.sign(np.ndarray.flatten(c_i, order='C') - offset_array) / (n_params * fidelity)
        
        return np.reshape(c_i, (m_events, n_params*fidelity))

    def convert_from_networkx(self, model):
        return dimod.BinaryQuadraticModel.from_networkx_graph(model)
    
    def create_sampler(self, endpoint_url, account_token):
        while cant_connect:
            try:
                self.sampler = DWaveSampler(endpoint=endpoint_url, token=account_token, solver=self.name)
                cant_connect = False
            except IOError:
                time.sleep(10)
                cant_connect = True
    
    def anneal(self, anneal_time=5):
        pass

# Independent function for simplifying problem
# -> should hold a couple simple pruning methods
def prune_variables(model):
    pass

# Independent function for quantum error correction
# -> should hold a single correction scheme to start (can always add)
def error_correction(model):
    pass




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