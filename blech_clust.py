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
import shutil

# Necessary blech_clust modules
from utils import read_file
from utils.blech_utils import entry_checker

############################################################

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

group_list = ['raw','raw_emg','digital_in','digital_out']
for this_group in group_list:
    if '/'+this_group in hf5:
        hf5.remove_node('/', this_group, recursive=True)
    hf5.create_group('/',this_group)
#hf5.create_group('/', 'raw')
#hf5.create_group('/', 'raw_emg')
#hf5.create_group('/', 'digital_in')
#hf5.create_group('/', 'digital_out')
hf5.close()
print('Created nodes in HF5')

# Create directories to store waveforms, spike times, clustering results, and plots
# And a directory for dumping files talking about memory usage in blech_process.py
# Check if dirs are already present, if they are, ask to delete and continue
# or abort
dir_list = ['spike_waveforms',
            'spike_times',
            'clustering_results',
            'Plots',
            'memory_monitor_clustering']
dir_exists = [x for x in dir_list if os.path.exists(x)]
recreate_msg = f'Following dirs are present :' + '\n' + f'{dir_exists}' + \
                '\n' + 'Overwrite dirs? (yes/y/n/no) ::: '

# If dirs exist, check with user
if len(dir_exists) > 0:
    recreate_str, continue_bool = entry_checker(\
            msg = recreate_msg,
            check_func = lambda x: x in ['y','yes','n','no'], 
            fail_response = 'Please enter (yes/y/n/no)')
# Otherwise, create all of them
else:
    continue_bool = True
    recreate_str = 'y'

# Break if user said n/no or gave exit signal
if continue_bool:
    if recreate_str in ['y','yes']:
        for x in dir_list:
            if os.path.exists(x):
                shutil.rmtree(x)
            os.makedirs(x)
    else:
        quit()
else:
    quit()

#os.mkdir('spike_waveforms')
#os.mkdir('spike_times')
#os.mkdir('clustering_results')
#os.mkdir('Plots')
#os.mkdir('memory_monitor_clustering')
print('Created dirs in data folder')

# Get the amplifier ports used
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()

## Pull out the digital input channels used, and convert them to integers
#dig_in = list(set(f[11:13] for f in file_list if f[:9] == 'board-DIN'))
#for i in range(len(dig_in)):
#	dig_in[i] = int(dig_in[i][0])
#dig_in.sort()

# Read dig-in data
# Pull out the digital input channels used, 
# and convert them to integers
dig_in_files = [x for x in file_list if "DIN" in x]
dig_in = [x.split('-')[-1].split('.')[0] for x in dig_in_files]
dig_in = sorted([int(x) for x in dig_in])

# Read the amplifier sampling rate from info.rhd - 
# look at Intan's website for structure of header files
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in}'

print(check_str)

with open(json_path[0], 'r') as params_file:
    info_dict = json.load(params_file)

all_car_group_vals = []
for region_name, region_elecs in info_dict['electrode_layout'].items():
    if not region_name == 'emg':
        for group in region_elecs:
            if len(group) > 0:
                all_car_group_vals.append(group)
all_electrodes = [electrode for region in all_car_group_vals \
                        for electrode in region]

emg_info = info_dict['emg']
emg_port = emg_info['port']
emg_channels = sorted(emg_info['electrodes'])


layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path) 

# Create arrays for each electrode
#read_file.create_hdf_arrays(hdf5_name, all_electrodes, 
#                            dig_in, emg_port, emg_channels)

# Read data files, and append to electrode arrays
if file_type[0] != 'one file per channel':
	print("Only files structured as one file per channel "
    "can be read at this time...")
    # Terminate blech_clust if something else has been used - to be changed later
	sys.exit() 

#read_file.read_files_abu(hdf5_name, dig_in, electrode_layout_frame) 
read_file.read_digins(hdf5_name, dig_in)
read_file.read_electrode_channels(hdf5_name, electrode_layout_frame)
if len(emg_channels) > 0:
    read_file.read_emg_channels(hdf5_name, electrode_layout_frame)

# Write out template params file to directory if not present
home_dir = os.getenv('HOME')
blech_clust_path = os.path.join(home_dir,'Desktop','blech_clust')
print(blech_clust_path)
params_template_path = os.path.join(
        blech_clust_path,
        'params/sorting_params_template.json')
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

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()

electrode_bool = electrode_layout_frame.loc[
        electrode_layout_frame.electrode_ind.isin(all_electrodes)]
not_none_bool = electrode_bool.loc[~electrode_bool.CAR_group.isin(["none","None",'na'])]
bash_electrode_list = not_none_bool.electrode_ind.values
job_count = np.min((len(bash_electrode_list), int(num_cpu-2)))
# todo: Account for electrodes labelled none when writing parallel command
runner_path = os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh') 
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel.sh'), 'w')
print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +\
        "--memfree 4G --retry-failed "+\
        f"--joblog {dir_name}/results.log "+\
        f"bash {runner_path} "+\
        #f"::: {{{','.join([str(x) for x in bash_electrode_list])}}}", 
        f"::: {' '.join([str(x) for x in bash_electrode_list])}", 
        file = f)
f.close()

# Then produce the file that runs blech_process.py
f = open(os.path.join(blech_clust_path,'blech_clust_jetstream_parallel1.sh'), 'w')
print("export OMP_NUM_THREADS=1", file = f)
blech_process_path = os.path.join(blech_clust_path,'blech_process.py')
print(f"python {blech_process_path} $1", file=f)
f.close()

# Dump the directory name where blech_process has to cd
f = open(os.path.join(blech_clust_path,'blech.dir'), 'w')
print(dir_name, file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')
