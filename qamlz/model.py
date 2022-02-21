import datetime

import numpy as np

from sklearn.metrics import accuracy_score

from .anneal import anneal, total_hamiltonian

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
        self.ngauges = [50, 10] + [1]*(n_iterations - 2)
        self.max_states = [16, 4] + [1]*(n_iterations - 2)
        self.nread = 200

        self.fix_vars = True
        self.prune_vars = anneal.default_prune
        self.cutoff = 95
        self.encode_vars = anneal.default_qac
        self.encoding_depth = 3      # from nested qac paper
        self.decode_vars = anneal.decode_qac
        self.encoded_energies = anneal.energies_qac
        self.encoded_uniques = anneal.uniques_qac

class Model:
    '''
        TODO: figure out how to save mus arrays, and ultimately mus files; add in validation data step
            and finish results dict
    '''
    def __init__(self, config, env):
        # add in hyperparameters in ModelConfig
        # -> this is where user determines how the model will train
        self.config = config
        self.env = env

        self.anneal_results = {}
        self.mus_dict = {}

    def train(self, env):
        mus = [np.zeros(np.size(env.C_i))]
        train_size = np.shape(env.X_train)[0]

        for i in range(self.config.n_iterations):
            new_mus = []
            for mu in mus:
                excited_states = anneal(self.config, i, env, mu)
                for excited_state in excited_states:
                    new_sigma = pow(self.config.zoom_factor, i) * self.config.zoom_factor
                    new_mu = mu + (pow(self.config.zoom_factor, i) * excited_state)
                    new_energy = total_hamiltonian(new_mu, new_sigma, env.C_i, env.C_ij) / (train_size - 1)
                    flips = np.ones(np.size(excited_state))
                    for qubit in range(len(excited_state)):
                        temp_s = np.copy(excited_state)
                        temp_s[qubit] = 0
                        old_energy = total_hamiltonian(mu, temp_s, new_sigma, env.C_i, env.C_ij) / (train_size - 1)
                        energy_diff = new_energy - old_energy
                        if energy_diff > 0:
                            flip_prob = self.config.flip_probs[i]
                            flip = np.random.choice([1, self.config.flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[qubit] = flip
                        else:
                            flip_prob = self.config.flip_others_probs[i]
                            flip = np.random.choice([1, self.config.flip_state], size=1, p=[1-flip_prob, flip_prob])[0]
                            flips[qubit] = flip
                    flipped_s = excited_state * flips
                    new_mus.append(mu + flipped_s*pow(self.config.zoom_factor, i)*self.config.zoom_factor)
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

    # split data up and run model on test data
    # return avg accuracy and std dev (from subsets)
    def evaluate(self, X_data, weights):
        ensemble_predictions = np.zeros(len(X_data[0]))
    
        return np.sign(np.dot(X_data.T, weights))