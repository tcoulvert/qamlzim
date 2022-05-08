import datetime

import numpy as np

from sklearn.metrics import accuracy_score

from .anneal import anneal
from .anneal import default_prune, default_qac, decode_qac

# Used to calculate the total hamiltonian of a certain problem
def total_hamiltonian(mu, s, sigma, C_i, C_ij):
    """
    Derived from Eq. 9 in QAML-Z paper (ZLokapa et al.)
    -> (Dot products of upper triangle)
    """
    ham = np.einsum("i, i", -C_i, s * sigma)
    ham = ham + np.einsum("i, i", np.einsum("ij, j", np.triu(C_ij, k=1), mu), s * sigma)
    ham = ham + np.einsum(
        "i, i", np.einsum("i, ij", s * sigma, np.triu(C_ij, k=1)), s * sigma
    )

    return ham


# Returns the ML algorithm's predictions
def evaluate(X_data, weights):
    """
    Simply computes the percentage of correct classifications on a given dataset.

    TODO: update so evaluates using the collapsed parameters?
          i dont remember exactly what this does, change?
    """
    return np.sign(np.dot(weights, X_data))


class ModelConfig:
    def __init__(self, n_iterations=10):
        """
        Configures the hyperparameters for the model. In essence, this controls how
        the model learns.

        TODO:

        Parameters (of the function):
        - n_iterations          Determines the number of repetitions for the training cycle.

        Configs (Hyperpararmeters of the model):
        - zoom_factor           Range(0,1) Controls the zooming rate per iteration. Essentially a
                                training step; larger zoom_factor means faster to train, but
                                slower to settle.
        - anneal_time           Controls the number of microseconds used by D-Wave machines;
                                larger anneal_time means more accurate results, but each run
                                takes more resources (D-Wave provides limited time).
        - flip_probs            Probailities of random spin-flips; used to prevent overtraining.
        - strengths             Scaling of the qubit (h) and coupling (J) weights.
        - max_states            Number of spin-states to record from D-Wave per iteration.
        - num_reads             Number of times to sample the final state from D-Wave
                                (shouldn't be lower than 100 or else the spin-state distribution
                                may not be properly recorded).
        - embedding             Stores the embedding across iterations to shorten runtime
                                (increases potential to fail early).
        - fix_vars              Defines whether to fix and remove low-variance qubits, as
                                described in D-Wave's fix_variables documentation.
        - prune_vars            Method used to remove weak couplings.
        - cutoff_percentile     Minimum percentile for couplings to NOT be removed.
        - encode_vars           Method used to encode qubits for error-correction
                                (encode_qac, encode_copy, or your own).
        - encoding_depth        Defined for NQAC as the number of copies, can be made into
                                something else for other error-correction schemes.
        - gamma                 Range(0,1] Defined for NQAC as the penalty associated with differing
                                spins for encoded-qubits corresponding to the same
                                logical qubit, can be made into something else for other
                                error-correction schemes.
        - decode_vars           Method used to decode qubits from error-correction
                                (decode_qac, decode_copy, or your own).
        """
        self.n_iterations = n_iterations
        self.zoom_factor = 0.5
        self.anneal_time = 5

        self.flip_probs = np.linspace(0.1, 0.01, num=n_iterations)

        self.strengths = [3.0, 1.0, 0.5, 0.2] + [0.1] * (n_iterations - 4)
        self.max_states = [16, 4] + [1] * (n_iterations - 2)
        self.num_reads = 200

        self.embedding = None

        self.fix_vars = True
        self.prune_vars = default_prune
        self.cutoff_percentile = 85
        self.encode_vars = default_qac
        self.encoding_depth = 3  # from nested qac paper
        self.gammas = np.linspace(  # from nested qac paper, defines strength of same logical-qubit couplings
            1, 0.1, num=10, endpoint=True
        )
        self.gamma = (
            1  # from nested qac paper, defines strength of same logical-qubit couplings
        )
        self.decode_vars = decode_qac


class Model:
    """
    Contains the model object (the output of the machine learning).

    TODO: finish results dict -> add in method for final test accuracy (AUROC) computation
    """

    def __init__(self, config, env):
        """
        Initializes the model object, all initialization parameters are contained in the config and env objects.

        TODO: add in hyperparameters in ModelConfig

        Parameters:
        - config            Config object that determines all training hyperparameters.
        - env               Env object that determines all data-processing hyperparameters.

        Model Vars:
        - config            Config object that determines all training hyperparameters.
        - env               Env object that determines all data-processing hyperparameters.
        - start_time        Records the time the model began training, useful for data storage.
        - anneal_results    Output dict of simple accuracies (accuracies computed with evaluate).
        - mus_dict          Output dict of optimized weightings (the actual classifier trained).
        """
        self.config = config
        self.env = env

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.anneal_results = {}
        self.mus_dict = {}

    def pick_excited_states(self, iter, excited_states, mu):
        """
        Selects spin states to keep dependent upon their associated energy.

        TODO:

        Parameters:
        - iter              Current iteration of the training.
        - excited_states    Obtained spins of a given read from D-Wave.
        - mu                Spins of qubits for this iteration.
        """
        new_mus = []
        for excited_state in excited_states:
            new_sigma = pow(self.config.zoom_factor, iter + 1)
            energy_flips = np.ones(np.size(excited_state))
            dwave_shift_energy = total_hamiltonian(
                mu, excited_state, new_sigma, self.env.C_i, self.env.C_ij
            )

            for idx in range(np.size(excited_state)):
                no_shift_state = np.copy(excited_state)
                no_shift_state[idx] = 0
                no_shift_energy = total_hamiltonian(
                    mu, no_shift_state, new_sigma, self.env.C_i, self.env.C_ij
                )
                anti_shift_state = np.copy(excited_state)
                anti_shift_state[idx] *= -1
                anti_shift_energy = total_hamiltonian(
                    mu, anti_shift_state, new_sigma, self.env.C_i, self.env.C_ij
                )

                if dwave_shift_energy < anti_shift_energy:
                    if dwave_shift_energy < no_shift_energy:
                        continue
                    else:
                        energy_flips[idx] = 0
                else:
                    if anti_shift_energy < no_shift_energy:
                        energy_flips[idx] = -1
                    else:
                        energy_flips[idx] = 0

            flip_prob = self.config.flip_probs[iter]
            overtrain_flips = np.random.choice(
                [-1, 1], size=np.size(energy_flips), p=[flip_prob, 1 - flip_prob]
            )
            new_state = excited_state * energy_flips * overtrain_flips
            new_mus.append(mu + new_state * new_sigma)

        return new_mus

    def train(self):
        """
        Performs the training of the ML model.

        TODO:
        """
        mus = [np.ones(np.size(self.env.C_i))]

        for iter in range(self.config.n_iterations):
            new_mus_arr = []
            for mu in mus:
                excited_states_arr = anneal(self.config, iter, self.env, mu)
                for excited_states in excited_states_arr:
                    new_mus_arr.append(
                        self.pick_excited_states(iter, excited_states, mu)
                    )
            accuracies = np.zeros(len(new_mus_arr))
            idx = 0
            for new_mus in new_mus_arr:
                avg_arr_val = []
                for new_mu in new_mus:
                    avg_arr_val.append(
                        accuracy_score(self.env.y_train, evaluate(new_mu, self.env.c_i))
                    )
                accuracies[idx] = np.mean(np.array(avg_arr_val))
                idx += 1
            chosen_new_mus = np.argmax(accuracies)
            self.anneal_results["iter%d" % iter] = accuracies[chosen_new_mus]
            mus = new_mus_arr[chosen_new_mus]
            mus_filename = "mus%05d-%d_iter%d_run%d__%s" % (
                self.env.train_size,
                self.env.fidelity,
                iter,
                chosen_new_mus,
                self.start_time,
            )
            self.mus_dict[mus_filename] = mus
