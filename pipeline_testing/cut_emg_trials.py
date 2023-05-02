# Import stuff!
import numpy as np
import sys
import os
from ..utils.blech_process_utils import path_handler
############################################################
# Figure out paths
path_handler = path_handler()
blech_clust_dir = path_handler.blech_clust_dir
############################################################
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
