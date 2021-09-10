#!/usr/bin/env python
import datetime
import subprocess

timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
anneal_cmd = './quantum_annealing.py'
anneal_nohup_cmd = 'nohup  &'



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
