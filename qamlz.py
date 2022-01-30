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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dwave.system.samplers import DWaveSampler

class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    def __init__(self, train_sig, train_bkg):
        self.train_sig = train_sig
        self.train_bkg = train_bkg

    def split_data():
        flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))/2
        flip_state = -1

class Model:
    def __init__(self, train_size=1000, n_folds=10, zoom_factor=0.5):
        # add in hyperparameters here
        # this is where user determines how the model will train
        self.train_size = train_size
        self.n_folds = n_folds
        self.zoom_factor = zoom_factor

    def evaluate(self, test_sig, test_bkg):
        # split data up and run model on test data
        # return avg accuracy and std dev (from subsets)
        
        return

    

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
