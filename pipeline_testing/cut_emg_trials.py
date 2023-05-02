# Import stuff!
import numpy as np
import sys
import os

# Use post-process sheet template to write out a new sheet for this dataset
script_path = os.path.dirname(os.path.realpath(__file__))
blech_clust_dir = os.path.dirname(script_path)
sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata

metadata_handler = imp_metadata(sys.argv)
os.chdir(metadata_handler.dir_name)
os.chdir('emg_output')

# Get filenames
# emg_data shape : channels x dig_ins x max_trials x duration 
# nonzero_trials shape : dig_ins x max_trials
filenames = ['emg_data.npy','nonzero_trials.npy']

# Chop down number of trials to have close to n total trials
total_trials = 10
data = [np.load(f) for f in filenames]
trials_per_digin = np.int(np.ceil(total_trials/data[0].shape[1]))
data[0] = data[0][:,:,:trials_per_digin]
data[1] = data[1][:,:trials_per_digin]

# Write back out
for i,f in enumerate(filenames):
    np.save(f,data[i])
