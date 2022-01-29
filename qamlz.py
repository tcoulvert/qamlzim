#!/usr/bin/env python3
import datetime
import json
import logging
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import time

from abc import ABC, abstractmethod
from dimod import BinaryQuadraticModel as BQM
from dwave.system.samplers import DWaveSampler
import networkx as nx


class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    def __init__(self, train_sig, train_bkg):
        self.train_sig = train_sig
        self.train_bkg = train_bkg

class Model:
    def __init__(self, train_size=1000, n_folds=10, zoom_factor=0.5, n_iterations=8, ):
        # add in hyperparameters here
        # this is where user determines how the model will train
        pass

    def evaluate(self, test_sig, test_bkg):
        # split data up and run model on test data
        # return avg accuracy and std dev (from subsets)
        
        return

# Allows for basis of many hardware classes 
# -> can be easily programmed to run on systems besides DWave
class Hardware(ABC):
    @abstractmethod
    def convert_from_networkx(self):
        pass

# Specific class for DWave hardware
class DWaveQA(Hardware,BQM):
    def __init__(self, name='Advantage_system4.2'):
        self.name = name
        self.sampler = None

    def convert_from_networkx(self, netx):
        return BQM.from_networkx_graph(netx)
    
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
    
        

