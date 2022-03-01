import datetime

import numpy as np

from sklearn.metrics import accuracy_score

from .anneal import anneal
from .anneal import default_prune, default_qac, decode_qac, energies_qac, uniques_qac

# Used to calculate the total hamiltonian of a certain problem
def total_hamiltonian(mu, s, sigma, C_i, C_ij):
    ''' Derived from Eq. 9 in QAML-Z paper (ZLokapa et al.)
        Dot products of upper triangle

        TODO: Check indecies; maybe add .T
    '''
    ham = np.einsum('i, i', -C_i, s*sigma)
    ham = ham + np.einsum('i, i', np.einsum('ij, j', np.triu(C_ij, k=1), mu), s*sigma)
    ham = ham + np.einsum('i, i', np.einsum('i, ij', s*sigma, np.triu(C_ij, k=1)), s*sigma)
    
    return ham

# Returns the ML algorithm's predictions
def evaluate(X_data, weights):
    '''
        TODO:
    '''
    return np.sign(np.dot(X_data, weights))

class ModelConfig:
    def __init__(self, n_iterations=10, zoom_factor=0.5):
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor
        self.anneal_time = 5
        
        self.flip_higher_probs = np.array([0.16, 0.08, 0.04, 0.02] + [0.01]*(n_iterations - 4))
        self.flip_lower_probs = self.flip_probs / 2

        self.strengths = [3.0, 1.0, 0.5, 0.2] + [0.1]*(n_iterations - 4)
        self.energy_fractions = [0.08, 0.04, 0.02] + [0.01]*(n_iterations - 3)
        self.ngauges = [50, 10] + [1]*(n_iterations - 2)
        self.max_states = [16, 4] + [1]*(n_iterations - 2)
        self.nread = 200

        self.embedding = None
        self.fixed_dict = None
        self.same_fixed_dict_counter = 0
        self.diff_fixed_dict_counter = 0

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

    def pick_excited_states(self, iter, excited_states, mu):
        for excited_state in excited_states:
            new_sigma = pow(self.config.zoom_factor, iter+1)
            new_energy = total_hamiltonian(mu, excited_state, new_sigma, self.env.C_i, self.env.C_ij)
            flips = np.ones(np.size(excited_state))
            for state in range(np.size(excited_state)):
                test_state = np.copy(excited_state)
                test_state[state] = 0
                test_energy = total_hamiltonian(mu, test_state, new_sigma, self.env.C_i, self.env.C_ij)
                if new_energy > test_energy:
                    flip_prob = self.config.flip_higher_probs[iter]
                else:
                    flip_prob = self.config.flip_lower_probs[iter]
                flip = np.random.choice([-1, 1], p=[flip_prob, 1-flip_prob])
                flips[state] = flip
            new_state = excited_state * flips

        return (mu + new_state*new_sigma)
            

    def train(self):
        '''
            TODO:
        '''
        mus = [np.ones(np.size(self.env.C_i))]

        for i in range(self.config.n_iterations):
            new_mus_arr = []
            for mu in mus:
                excited_states_arr = anneal(self.config, i, self.env, mu)
                for excited_states in excited_states_arr:
                    new_mus_arr.append([self.pick_excited_states(i, excited_states, mu)])
            accuracies = np.zeros(len(new_mus_arr))
            for new_mus in range(len(new_mus_arr)):
                avg_arr_val =[]
                for new_mu in new_mus:
                    avg_arr_val.append(accuracy_score(self.env.y_train, evaluate(new_mu, self.env.c_i)))
                np.append(accuracies, np.mean(np.array(avg_arr_val)))
            chosen_new_mus = np.argmax(accuracies)
            mus = new_mus_arr[chosen_new_mus]
            mus_filename = 'mus%05d_iter%d_run%d-%s.npy' % (self.env.train_size, i, chosen_new_mus, self.start_time)
            self.mus_dict[mus_filename] = np.array(mus)

