"""
Initialize directory for emg processing (regardless of spike sorting)

If HDF5 present, use that, if not present, create new one
"""

# Necessary python modules
import easygui
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd

# Necessary blech_clust modules
sys.path.append('..')
from utils import read_file

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

print(f'Processing : {dir_name}')

# Get the type of data files (.rhd or .dat)
file_type = ['one file per channel']

# Change to that directory
os.chdir(dir_name)

# Check that experimental_info json file is present
# If not present, refuse to cooperate
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))
if len(json_path) == 0:
    raise Exception('Must have experimental info json before proceeding \n'\
            'Run blech_exp_info.py first \n'\
            '== Exiting ==')
    exit()

# Get the names of all files in this directory
file_list = os.listdir('./')

# Create hdf5 file, and make groups for raw data, raw emgs, 
# digital outputs and digital inputs, and close

# Grab directory name to create the name of the hdf5 file
# If HDF5 present, use that, otherwise, create new one
h5_search = glob.glob('*.h5')
if len(h5_search):
    hdf5_name = h5_search[0] 
    print(f'HDF5 file found...Using file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'r+')
else:
    hdf5_name = str(os.path.dirname(dir_name)).split('/')[-1]+'.h5'
    print(f'No HDF5 found...Creating file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'w', title = hdf5_name[-1])

# Remove any previous raw_emg data
if '/raw_emg' in hf5:
    hf5.remove_node('/','raw_emg', recursive=True)
# Create raw_emg group in HDF5 file
hf5.create_group('/', 'raw_emg')

if '/digital_in' in hf5:
    hf5.remove_node('/','digital_in', recursive=True)
# Create raw_emg group in HDF5 file
hf5.create_group('/', 'digital_in')
print('Created nodes in HF5')
hf5.close()

# Get the amplifier ports used
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()

with open(json_path[0], 'r') as params_file:
    info_dict = json.load(params_file)

emg_info = info_dict['emg']
emg_port = emg_info['port']
emg_channels = sorted(emg_info['electrodes'])

layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path) 

# Read dig-in data
# Pull out the digital input channels used, and convert them to integers
dig_in = list(set(f[11:13] for f in file_list if f[:9] == 'board-DIN'))
for i in range(len(dig_in)):
	dig_in[i] = int(dig_in[i][0])
dig_in.sort()

read_file.read_digins(hdf5_name, dig_in)
read_file.read_emg_channels(hdf5_name, electrode_layout_frame)

# Write out template params file to directory if not present
# Read the amplifier sampling rate from info.rhd - 
# look at Intan's website for structure of header files
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])

home_dir = os.getenv('HOME')
params_template_path = os.path.join(home_dir,'Desktop/blech_clust/params/sorting_params_template.json')
params_template = json.load(open(params_template_path,'r'))
# Info on taste digins and laser should be in exp_info file
all_params_dict = params_template.copy() 
all_params_dict['sampling_rate'] = sampling_rate

params_out_path = hdf5_name.split('.')[0] + '.params'
if not os.path.exists(params_out_path):
    print('No params file found...Creating new params file')
    with open(params_out_path, 'w') as params_file:
        json.dump(all_params_dict, params_file, indent = 4)
else:
    print("Params file already present...not writing a new one")
