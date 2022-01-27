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
import read_file

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

print(f'Processing : {dir_name}')
#cont = 'a'
#while cont not in ['y','n']:
#    cont = input('Is this the correct directory (y/n): \n{}\n::'.format(dir_name))
#if cont == 'n':
#    sys.exit('Incorrect dir')

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

# Grab directory name to create the name of the hdf5 file
hdf5_name = str(os.path.dirname(dir_name)).split('/')

# Create hdf5 file, and make groups for raw data, raw emgs, 
# digital outputs and digital inputs, and close
hf5 = tables.open_file(hdf5_name[-1]+'.h5', 'w', title = hdf5_name[-1])
hf5.create_group('/', 'raw')
hf5.create_group('/', 'raw_emg')
hf5.create_group('/', 'digital_in')
hf5.create_group('/', 'digital_out')
hf5.close()
print('Created nodes in HF5')

# Create directories to store waveforms, spike times, clustering results, and plots
os.mkdir('spike_waveforms')
os.mkdir('spike_times')
os.mkdir('clustering_results')
os.mkdir('Plots')
print('Created dirs in data folder')

# Get the amplifier ports used
ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
# Sort the ports in alphabetical order
ports.sort()

# Pull out the digital input channels used, and convert them to integers
dig_in = list(set(f[11:13] for f in file_list if f[:9] == 'board-DIN'))
for i in range(len(dig_in)):
	dig_in[i] = int(dig_in[i][0])
dig_in.sort()

# Read the amplifier sampling rate from info.rhd - 
# look at Intan's website for structure of header files
sampling_rate = np.fromfile('info.rhd', dtype = np.dtype('float32'))
sampling_rate = int(sampling_rate[2])

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in}'

print(check_str)

# Get the emg electrode ports and channel numbers from the user
# If only one amplifier port was used in the experiment, that's the emg_port. 
# Else ask the user to specify
emg_port = ''
#if len(ports) == 1:
#	emg_port = list(ports[0])
#else:
#	emg_port = easygui.multchoicebox(\
#        msg = 'Which amplifier port were the EMG electrodes hooked up to? '\
#        'Just choose any amplifier port if you did not hook up an EMG at all.', 
#        choices = tuple(ports))
## Now get the emg channel numbers, and convert them to integers
#emg_channels = easygui.multchoicebox(\
#        msg = 'Choose the channel numbers for the EMG electrodes. 
#        Click clear all and ok if you did not use an EMG electrode', 
#        choices = tuple([i for i in range(32)]))

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
read_file.create_hdf_arrays(hdf5_name[-1]+'.h5', all_electrodes, 
                            dig_in, emg_port, emg_channels)

# Read data files, and append to electrode arrays
if file_type[0] == 'one file per channel':
	read_file.read_files_abu(hdf5_name[-1]+'.h5', dig_in, electrode_layout_frame) 
else:
	print("Only files structured as one file per channel can be read at this time...")
	sys.exit() # Terminate blech_clust if something else has been used - to be changed later

# And print them to a blech_params file
clustering_params = {'max_clusters' : 7,
                    'num_iter' : 1000, 
                    'thresh' : 0.0001,
                    'num_restarts' : 10}
data_params = {'voltage_cutoff' : 10000,
                'max_breach_rate' : 1,
                'max_secs_above_cutoff' : 60,
                'max_mean_breach_rate_persec' : 100,
                'wf_amplitude_sd_cutoff' : 3}
bandpass_params = {'bandpass_lower_cutoff' : 300,
                    'bandpass_upper_cutoff' : 3000}
spike_snapshot = {'spike_snapshot_before' : 1,
                    'spike_snapshot_after' : 1.5}
psth_params = {'psth_params' : 
                    {'durations' : [500,2000],
                        'window_size' : 250,
                        'step_size' : 25}}
pal_iden_calc_params = {'pal_iden_calc_params' : {
                    'window_size' : 250,
                    'step_size' : 25}}
discrim_analysis_params = {'discrim_analysis_params' : {
                    'bin_num' : 4,
                    'bin_width' : 500,
                    'p-value' : 0.05}}

# Info on taste digins and laser should be in exp_info file
all_params_dict = {**clustering_params, **data_params,
                **bandpass_params, **spike_snapshot,
                **psth_params,
                'sampling_rate' : sampling_rate,
                'similarity_cutoff' : 50,
                'spike_array_durations' : [2000,5000],
                **pal_iden_calc_params,
                **discrim_analysis_params,
                'palatability_window' : [700,1200]}

with open(hdf5_name[-1]+'.params', 'w') as params_file:
    json.dump(all_params_dict, params_file, indent = 4)

# Make a directory for dumping files talking about memory usage in blech_process.py
os.mkdir('memory_monitor_clustering')

# Ask for the HPC queue to use - was in previous version, now just use all.q

# Grab Brandeis unet username
username = ['abuzarmahmood']

# Dump shell file for running array job on the user's blech_clust folder on the desktop
os.chdir('/home/%s/Desktop/blech_clust' % username[0])
f = open('blech_clust.sh', 'w')
print("export OMP_NUM_THREADS=1", file = f)
print("cd /home/%s/Desktop/blech_clust" % username[0], file=f)
print("python blech_process.py", file=f)
f.close()

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()
# Then produce the file generating the parallel command
# If EMG is present, don't add EMG electrode to list of electrodes
# to be processed
# Check if EMG present
# Write appropriate electrodes to file

# Electrode + 1 because blech_process does -1
f = open('blech_clust_jetstream_parallel.sh', 'w')
print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed "\
        "--joblog {:s}/results.log bash blech_clust_jetstream_parallel1.sh ::: {{{}}}"\
        .format(int(num_cpu//4), dir_name, ",".join([str(x) for x in all_electrodes]))
        , file = f)
f.close()

#else:
#    f = open('blech_clust_jetstream_parallel.sh', 'w')
#    print("parallel -k -j {:d} --noswap --load 100% --progress --memfree 4G --retry-failed "\
#            "--joblog {:s}/results.log bash blech_clust_jetstream_parallel1.sh ::: {{1..{:d}}}"\
#            .format(int(num_cpu//4), dir_name, int(len(ports)*32-len(emg_channels)))
#            , file = f)
#    f.close()

# Then produce the file that runs blech_process.py
f = open('blech_clust_jetstream_parallel1.sh', 'w')
print("export OMP_NUM_THREADS=1", file = f)
print("python blech_process.py $1", file = f)
f.close()

# Dump the directory name where blech_process has to cd
f = open('blech.dir', 'w')
print(dir_name, file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')
