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

class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    # X_train, y_train should be formatted like scikit-learn data arrays are
    # -> X_train is the train data, y_train is the train labels
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = None
        self.y_validation = None

    # Splits training data into train and validation 
    # -> allows hyperparameter adjustment
    def create_validation_data(self, n_splits=10, validation_pct=0.2, random_seed=None):
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=validation_pct, random_state=random_seed)

        # X_tra, X_val = np.empty((1,0)), np.empty((1,0))
        # y_tra, y_val = np.empty((1,0)), np.empty((1,0))
        # i = 0
        for train_indices, validation_indices in sss.split(self.X_train, self.y_train):
            pass
            # X_tra[i], X_val[i] = np.append(X_tra[i], self.X_train[train_indices]), np.append(X_val[i], self.X_train[validation_indices])
            # y_tra, y_val = np.append(y_tra, self.y_train[train_indices]), np.append(y_val, self.y_train[validation_indices])
            # i += 1

        # print('X_tra:', X_tra.shape, 'X_val:', X_val, 'y_tra:', y_tra, 'y_val:', y_val)
        # 
        # train_sig, validation_sig = np.empty(), np.empty()
        # train_bkg, validation_bkg = np.empty(), np.empty()
        # for train_sig_indices, validation_sig_indices in sss.split(np.zeros(self.train_sig.size), self.train_sig):
        #     train_sig.append(self.train_sig[train_sig_indices])
        #     validation_sig.append(self.train_sig[validation_sig_indices])
        # for train_bkg_indices, validation_bkg_indices in sss.split(np.zeros(self.train_bkg.size), self.train_bkg):
        #     train_bkg.append(self.train_bkg[train_bkg_indices])
        #     validation_bkg.append(self.train_bkg[validation_bkg_indices])
        # self.train_sig, self.validation_sig = train_sig, validation_sig
        # self.train_bkg, self.validation_bkg = train_bkg, validation_bkg
        

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
    def __init__(self, n_iterations=10, zoom_factor=0.5):
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor

        self.flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        self.flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))/2
        FLIP_STATE = -1
        

# Allows for basis of many hardware classes 
# -> can be easily programmed to run on systems besides DWave
class Hardware(abc.ABC):
    @abc.abstractmethod
    def convert_from_networkx(self):
        pass

# Specific class for DWave hardware
class DWaveQA(Hardware,dimod.BinaryQuadraticModel):
    def __init__(self, name='Advantage_system4.2'):
        self.name = name
        self.sampler = None

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

# Indepdendent function the pre- and post-processing on the input data
# -> should do the folding and iterating
def data_processing(sig, bkg):
    pass

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])

env = TrainEnv(X, y)
env.create_validation_data(n_splits=5, validation_pct=0.5, random_seed=0)