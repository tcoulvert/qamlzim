#!/usr/bin/env python3
import datetime
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from scipy.integrate import simps
from scipy.interpolate import interp1d

script_path = os.path.dirname(os.path.realpath(__file__))
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

train_sizes = [4000]
n_folds = 5 # Must be same as n_iterations from run

AUGMENT_SIZE = 7   # must be an odd number (since augmentation includes original value in middle)
AUGMENT_OFFSET = 0.0225 / AUGMENT_SIZE

POISSON = True
b_start = 1000000
b_end = 2000000

rng_one = np.random.default_rng(0)
rng_two = np.random.default_rng(0)

# Change to easily store data for analysis in annealyze AND jupyter notebook
auroc_results = {
    'train_sizes': train_sizes,
    'AUGMENT_SIZE': AUGMENT_SIZE,
    'AUGMENT_OFFSET': AUGMENT_OFFSET,
    'POISSON': POISSON,
    'data': []
}

# Add in a class to parse the json data from q_a file

def create_augmented_data(sig, bkg):
    offset = AUGMENT_OFFSET
    scale = AUGMENT_SIZE

    n_samples = len(sig) + len(bkg)
    n_classifiers = sig.shape[1]
    predictions_raw = np.concatenate((sig, bkg))
    predictions_raw = np.transpose(predictions_raw)
    predictions = np.zeros((n_classifiers * scale, n_samples))
    for i in range(n_classifiers):
        for j in range(scale):
            predictions[i*scale + j] = np.sign(predictions_raw[i] + (j-scale//2)*offset) / (n_classifiers * scale)
    y = np.concatenate((np.ones(len(sig)), -np.ones(len(bkg))))
    return predictions, y

def ensemble(predictions, weights):
    n_classifiers = len(weights)
    return np.dot(predictions.T, weights)

def auc(predictions, y_test, test_sig, test_bkg, poisson):
    cutoff = -1.0
    increments = 1000
    bkg_rejection = [0.0]
    sig_efficiency = [1.0]
    predictions /= np.amax(np.abs(predictions))
    
    for j in range(increments+1):
        cutoff += j * (2.0/increments)
        cut_predictions = np.sign(predictions - cutoff)
        agree = (y_test == cut_predictions).astype(np.float64)
        agree *= poisson
        se = np.sum(agree[np.where(y_test == 1.0)]) / len(test_sig)
        br = np.sum(agree[np.where(y_test == -1.0)]) / len(test_bkg)
        if br > 0 and se < sig_efficiency[-1] and br > bkg_rejection[-1] and se > 0:
            sig_efficiency.append(se)
            bkg_rejection.append(br)
        elif se == 0.0:
            break
    
    # add far term to make up for poisson
    bkg_rejection.append(2.0)
    sig_efficiency.append(0.0)
    
    bkg_rejection = np.array(bkg_rejection)
    sig_efficiency = np.array(sig_efficiency)
    
    sort = np.argsort(bkg_rejection)
    bkg_rejection = bkg_rejection[sort]
    sig_efficiency = sig_efficiency[sort]
    
    return bkg_rejection, sig_efficiency

def rand_delete(remaining_val, num_samples, train_data=False, test_data=False):
    # Potentially want to return array of sampled indeces, left in for convenience
    # picked_indeces = np.array()
    picked_values = np.array(0)
    for i in range(int(num_samples)):
        if train_data:
            picked_index = rng_one.integers(0, len(remaining_val))
        elif test_data:
            picked_index = rng_two.integers(0, len(remaining_val))
        # picked_indeces = np.append(picked_index)
        picked_values = np.append(picked_values, remaining_val[picked_index])
        remaining_val = np.delete(remaining_val, picked_index)
    
    return picked_values

def make_output_file(failnote=''):
    filename = '%sauroc_accuracy_results-%s.json' % (failnote, timestamp)
    destdir = os.path.join(script_path, 'qamlz_auroc')
    filepath = os.path.join(destdir, filename)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    json.dump(auroc_results, open(filepath, 'w'), indent=4)

def load_json():
    file_found = False
    failnote = ''
    while not file_found:
        try:
            filename = '%sanneal_results-%s' % (failnote, timestamp)
            file_found = True
        except FileNotFoundError:
            if failnote == '':
                failnote = 'FAILED__'
                continue
            elif failnote == 'FAILED__':
                print('File does not exist')
                sys.exit(1)
    return json.load(filename)

def main(sig=None, bkg=None, mus_dir=None, mus_filenames=[], train_size=None):
    n_folds = len(mus_filenames)
    if sig is None and bkg is None:
        sig = np.loadtxt('sig.csv')
        bkg = np.loadtxt('bkg.csv')
    sig_pct = float(len(sig)) / (len(sig) + len(bkg))
    bkg_pct = float(len(bkg)) / (len(sig) + len(bkg))
    print('loaded data')

    if POISSON:
        poisson_runs = 5
    else:
        poisson_runs = 1
    for i in range(len(train_sizes)):
        # train_size = train_sizes[i]
        # print('training with size', train_size)
        sig_indices = np.arange(len(sig))
        bkg_indices = np.arange(len(bkg))
        
        remaining_sig = sig_indices
        remaining_bkg = bkg_indices
        
        cts = np.zeros(n_folds*poisson_runs)
        
        for f in range(n_folds):
            run = 0
            if f == 1:
                run = 10
            elif f == 2:
                run = 1
            print('fold', f)
            train_sig = rand_delete(remaining_sig, sig_pct*train_size, train_data=True)
            train_bkg = rand_delete(remaining_bkg, bkg_pct*train_size, train_data=True)
            
            test_sig = np.delete(sig_indices, train_sig)
            test_bkg = np.delete(bkg_indices, train_bkg)

            valid_sig = rand_delete(test_sig, sig_pct*train_size, test_data=True)
            valid_bkg = rand_delete(test_bkg, bkg_pct*train_size, test_data=True)

            predictions_test, y_test = create_augmented_data(sig[test_sig], bkg[test_bkg])
            predictions_valid, y_valid = create_augmented_data(sig[valid_sig], bkg[valid_bkg])
            if mus_filenames is None:
                mus_filename = 'FAILmus04000-7_iter%d_run%d__2022-03-25_15-18-47.npy' % (f, run)
            else:
                mus_filename = mus_filenames[f] + '.npy'
            print(mus_filename)

            if mus_dir is None:
                mus_destdir = os.path.join(script_path, '../qamlz_for_higgs/mus/2022-03-25_15-18-47/')
            else:
                mus_destdir = mus_dir
            mus_filepath = (os.path.join(mus_destdir, mus_filename))
            excited_weights = np.load(mus_filepath)
            
            for p in range(poisson_runs):
                if POISSON:
                    poisson = np.random.poisson(1.0, len(y_test))
                    valid_poisson = np.random.poisson(1.0, len(y_valid))
                else:
                    poisson = np.ones(len(y_test))
                
                bkg_grid = np.linspace(0, 1.05, num=1000)
                sig_efficiencies = np.zeros((len(excited_weights), len(bkg_grid)))
                valid_sig_efficiencies = np.zeros((len(excited_weights), len(bkg_grid)))
                for w in range(len(excited_weights)):
                    continuous_predictions = ensemble(predictions_test, excited_weights[w])
                    bkg_rejection, sig_efficiency = auc(continuous_predictions, y_test, test_sig, test_bkg, poisson)
                    interp = interp1d(bkg_rejection, sig_efficiency, kind='cubic')
                    
                    valid_continuous_predictions = ensemble(predictions_valid, excited_weights[w])
                    valid_bkg_rejection, valid_sig_efficiency = auc(valid_continuous_predictions, y_valid, valid_sig, valid_bkg, valid_poisson)
                    valid_interp = interp1d(valid_bkg_rejection, valid_sig_efficiency, kind='cubic')
                    
                    sig_efficiencies[w] = interp(bkg_grid)
                    valid_sig_efficiencies[w] = valid_interp(bkg_grid)
                    for b in range(len(bkg_grid)):
                        if bkg_grid[b] > bkg_rejection[-2]:
                            sig_efficiencies[w][b] = 0
                        if bkg_grid[b] > valid_bkg_rejection[-2]:
                            valid_sig_efficiencies[w][b] = 0
                
                # take supremum among signal efficiency curves
                sig_efficiency_ind = np.argmax(valid_sig_efficiencies, axis=0)
                sig_efficiency_max = sig_efficiencies[(sig_efficiency_ind, np.arange(len(bkg_grid)))]
                continuous_auc = simps(sig_efficiency_max, bkg_grid)
                print('fold auc', continuous_auc)
                
                cts[f*poisson_runs + p] = continuous_auc
        
        mean = np.mean(cts)
        std = np.std(cts)
        auroc_results['data'].append({'train_size': train_size, 'mean': mean, 'std_dev': std})
        print('auroc mean', mean)
        print('auroc stdev', std)
    make_output_file()

def help():
    print('options:')
    print('  --help     Show this message')

if __name__ == '__main__':
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--help':
            help()
        elif arg == '--timestamp':
            i += 1
            timestamp = str(sys.argv[i])
        elif arg == '--json':
            i += 1
            anneal_results = load_json()
            GIT_HASH = anneal_results['git hash']
            timestamp = anneal_results['timestamp']
            dwave_architecture = anneal_results['architecture']
            FIXING_VARIABLES = anneal_results['git hash']
            a_time = anneal_results['anneal time']
            train_sizes = anneal_results['train sizes']
            n_folds = anneal_results['number of data folds']
            zoom_factor = anneal_results['zoom factor']
            n_iterations = anneal_results['number of iterations']
            AUGMENT_CUTOFF_PERCENTILE = anneal_results['augment cutoff percentile']
            AUGMENT_SIZE = anneal_results['number of classifiers']
            AUGMENT_OFFSET = anneal_results['classifier offset']
            mus_files = anneal_results['mus arrays']
            error_check_arr = anneal_results['errors']
        else:
            print('Unrecognized option %s' % (arg))
            sys.exit(1)
        i += 1
    main()