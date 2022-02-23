import datetime

import numpy as np

from sklearn.metrics import accuracy_score

from .anneal import anneal, total_hamiltonian
from .anneal import default_prune, default_qac, decode_qac, energies_qac, uniques_qac

class ModelConfig:
    def __init__(self, n_iterations=10, zoom_factor=0.5):
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor
        self.anneal_time = 5
        
        self.flip_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        self.flip_others_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4)) / 2

        self.strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        self.energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        self.ngauges = [50, 10] + [1]*(n_iterations - 2)
        self.max_states = [16, 4] + [1]*(n_iterations - 2)
        self.nread = 200

        self.embedding = None

        self.fix_vars = True
        self.prune_vars = default_prune
        self.cutoff = 95
        self.encode_vars = default_qac
        self.encoding_depth = 3      # from nested qac paper
        self.decode_vars = decode_qac
        self.encoded_energies = energies_qac
        self.encoded_uniques = uniques_qac

class Model:
    '''
        TODO: finish results dict -> add in method for final test accuracy (AUROC) computation
    '''
    def __init__(self, config, env):
        # add in hyperparameters in ModelConfig
        # -> this is where user determines how the model will train
        self.config = config
        self.env = env

        self.start_time = datetime.datetime.now().strftime('%Y/%m/%d__%H-%M-%S')
        self.anneal_results = {}
        self.mus_dict = {}

    def pick_excited_states(config, env, iter, excited_states, mu, train_size):
        for excited_state in excited_states:
            new_sigma = pow(config.zoom_factor, iter+1)
            new_mu = mu + (pow(config.zoom_factor, iter) * excited_state)
            new_energy = total_hamiltonian(new_mu, new_sigma, env.C_i, env.C_ij) / (train_size - 1)
            flips = np.ones(np.size(excited_state))
            for qubit in range(np.size(excited_state)):
                temp_s = np.copy(excited_state)
                temp_s[qubit] = 0
                old_energy = total_hamiltonian(mu, temp_s, new_sigma, env.C_i, env.C_ij) / (train_size - 1)
                energy_diff = new_energy - old_energy
                if energy_diff > 0:
                    flip_prob = config.flip_probs[iter]
                    flip = np.random.choice([-1, 1], size=1, p=[1-flip_prob, flip_prob])[0]
                    flips[qubit] = flip
                else:
                    flip_prob = config.flip_others_probs[iter]
                    flip = np.random.choice([-1, 1], size=1, p=[1-flip_prob, flip_prob])[0]
                    flips[qubit] = flip
            flipped_s = excited_state * flips

        return (mu + flipped_s*new_sigma)
            

    def train(self):
        mus = [np.zeros(np.size(self.env.C_i))]

        for i in range(self.config.n_iterations):
            new_mus = []
            for mu in mus:
                excited_states_arr = anneal(self.config, i, self.env, mu)
                for j in range(np.size(excited_states_arr, 0)):
                    new_mus.append([self.pick_excited_states(self.config, self.env, i, excited_states_arr[j], mu, np.shape(self.env.X_train)[0])])
            accuracies = np.zeros(len(new_mus))
            for j in range(len(new_mus)):
                avg_arr_val =[]
                for mu in new_mus[j]:
                    avg_arr_val.append(accuracy_score(self.env.y_val, self.evaluate(self.env.X_val, mu)))
                np.append(accuracies, np.mean(np.array(avg_arr_val)))
            mus = new_mus[np.where(np.max(accuracies))]
            mus_filename = 'mus%05d_iter%d_run%d-%s.npy' % (np.shape(self.env.X_train)[0], i, j, self.start_time)
            self.mus_dict[mus_filename] = np.array(mus)

    # Returns the ML algorithm's predictions
    def evaluate(self, X_data, weights):
        return np.sign(np.dot(X_data.T, weights))