"""
Script to reload raw_emg if it gets deleted without going through blech_clust.py
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
from tqdm import tqdm

# Necessary blech_clust modules
sys.path.append('../..')
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
json_path = glob.glob(os.path.join(dir_name, '*.info'))
if len(json_path) == 0:
    raise Exception('Must have experimental info json before proceeding \n'\
            'Run blech_exp_info.py first \n'\
            '== Exiting ==')
    exit()

# Get the names of all files in this directory
file_list = os.listdir('./')

# Open HDF5 file
hdf5_name = glob.glob(os.path.join(dir_name, "*.h5"))[0]
hf5 = tables.open_file(hdf5_name, 'r+')
# Remove any previous raw_emg data
if '/raw_emg' in hf5:
    hf5.remove_node('/','raw_emg', recursive=True)
# Create raw_emg group in HDF5 file
hf5.create_group('/', 'raw_emg')

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

# Read EMG data from amplifier channels
atom = tables.IntAtom()
emg_counter = 0
#for port in ports:
for num,row in tqdm(electrode_layout_frame.iterrows()):
    # Loading should use file name 
    # but writing should use channel ind so that channels from 
    # multiple boards are written into a monotonic sequence
    if 'emg' in row.CAR_group.lower():
        print(f'Reading : {row.filename, row.CAR_group}')
        port = row.port
        channel_ind = row.electrode_ind
        data = np.fromfile(row.filename, dtype = np.dtype('int16'))
        el = hf5.create_earray('/raw_emg', f'emg{emg_counter:02}', atom, (0,))
        exec(f"hf5.root.raw_emg.emg{emg_counter:02}."\
                "append(data[:])")
        emg_counter += 1
        hf5.flush()

hf5.close()
