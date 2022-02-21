import time

import dimod
import dwave as dw
import minorminer
import networkx as nx
import numpy as np
import sklearn as sk

from dwave.system.samplers import DWaveSampler

import anneal
import model

# Numpy arrays should be row-major for best performance

class TrainEnv:
    # require for the data to be passed as NP data arrays 
    # -> clear to both user and code what each array is
    # X_train, y_train should be formatted like scikit-learn data arrays are
    # -> X_train is the train data, y_train is the train labels
    def __init__(self, X_train, y_train, endpoint_url, account_token, X_val=None, y_val=None, fidelity=7):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.fidelity = fidelity
        self.fidelity_offset = 0.0225 / fidelity

        self.C_i = None
        self.C_ij = None
        self.data_preprocess()

        self.sampler = None
        self.create_sampler(endpoint_url, account_token)

    def create_sampler(self, endpoint_url, account_token, name='Advantage_system4.2'):
        while cant_connect:
            try:
                self.sampler = DWaveSampler(endpoint=endpoint_url, token=account_token, solver=name)
                cant_connect = False
            except IOError:
                time.sleep(10)
                cant_connect = True

    # Splits training data into train and val 
    # -> val allows hyperparameter adjustment
    def sklearn_data_wrapper(self, sk_model):
        X_tra, X_val = [], []
        y_tra, y_val = [], []
        for train_indices, val_indices in sk_model.split(self.X_train, self.y_train):
            X_tra.append(self.X_train[train_indices]), X_val.append(self.X_train[val_indices])
            y_tra.append(self.y_train[train_indices]), y_val.append(self.y_train[val_indices])
        
        self.X_train, self.X_val = np.array(X_tra), np.array(X_val)
        self.y_train, self.y_val = np.array(y_tra), np.array(y_val)

    def data_preprocess(self):
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
        m_events, n_params = np.shape(self.X_train) # [M events (rows) x N parameters (columns)]

        c_i = np.repeat(self.X_train, repeats=self.fidelity, axis=1) # [M events (rows) x N*fidelity parameters (columns)]
        
        offset_array = self.fidelity_offset * (np.tile(np.arange(self.fidelity), m_events * n_params) - self.fidelity//2)
        c_i = np.sign(np.ndarray.flatten(c_i, order='C') - offset_array) / (n_params * self.fidelity)
        c_i = np.reshape(c_i, (m_events, n_params*self.fidelity))

        C_i = np.dot(self.y_train, c_i)
        C_ij = np.einsum('ij,kj', c_i, c_i)

        self.C_i, self.C_ij = C_i, C_ij