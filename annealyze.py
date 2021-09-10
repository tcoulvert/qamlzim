#!/usr/bin/env python
import datetime
import subprocess

timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
platform = 'PEGASUS'
fixing_vars = True
a_time = 5
train_sizes = '[100,1000,5000,10000,15000,20000]'
n_folds = 10
zoom_factor = 0.5
n_iterations = 8
cutoff_percentile = 85
augment_size = 9
# augment_offset = 0.0225 / augment_size

anneal_cmd = './quantum_annealing.py --timestamp %s --architecture %s --fixing_vars %d --augment_time %d --train_sizes %s --n_folds %d --zoom_factor %f --n_iterations %d --cutoff_percentile %d --augment_size %d' % (timestamp, platform, fixing_vars, a_time, train_sizes, n_folds, zoom_factor, n_iterations, cutoff_percentile, augment_size)
anneal_nohup_cmd = 'nohup ./quantum_annealing.py --timestamp %s --architecture %s --augment_time %d --train_sizes %s --n_folds %d --zoom_factor %f --n_iterations %d --cutoff_percentile %d --augment_size %d &' % (timestamp, platform, a_time, train_sizes, n_folds, zoom_factor, n_iterations, cutoff_percentile, augment_size)

def anneal():
    subprocess.call(anneal_cmd, SHELL=True)
def anneal_nohup():
    subprocess.call(anneal_nohup_cmd, SHELL=True)


# Below is an example of how you could instead run the files via import
# import quantum_annealing

# quantum_annealing.AUGMENT_SIZE = 7
# ... 
# (the rest of the state/global parameters) 
# ...
# quantum_annealing.init()
# quantum_annealing.main()
#
# roc_excited_states.AUGMENT_SIZE = 7
# ... 
# (the rest of the state/global parameters) 
# ...
# roc_excited_states.main()
