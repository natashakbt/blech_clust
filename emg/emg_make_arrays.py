# TODO: This code doesn't allow for uneven numbers of trials
# TODO: Replace exec statements

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import glob
import json
import re

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

os.chdir(dir_name)
# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Grab the names of the arrays containing digital inputs, 
# and pull the data into a numpy array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
    dig_in_pathname.append(node._v_pathname)
    exec("dig_in.append(hf5.root.digital_in.%s[:])" 
            % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# Get the stimulus delivery times - 
# take the end of the stimulus pulse as the time of delivery
dig_on = []
for i in range(len(dig_in)):
    dig_on.append(np.where(dig_in[i,:] == 1)[0])

start_points = []
end_points = []
for on_times in dig_on:
        start = []
        end = []
        try:
                # Get the start of the first trial
                start.append(on_times[0]) 
        except:
                # Continue without appending anything if this port wasn't on at all
                pass 
        for j in range(len(on_times) - 1):
                if np.abs(on_times[j] - on_times[j+1]) > 30:
                        end.append(on_times[j])
                        start.append(on_times[j+1])
        try:
                # append the last trial which will be missed by this method
                end.append(on_times[-1]) 
        except:
                # Continue without appending anything if this port wasn't on at all
                pass 
        start_points.append(np.array(start))
        end_points.append(np.array(end))

# Extract taste dig-ins from experimental info file
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)

dig_in_channel_nums = info_dict['taste_params']['dig_ins']
dig_in_channels = [dig_in_pathname[i] for i in dig_in_channel_nums]
#dig_in_channel_inds = np.arange(len(dig_in_channels))
dig_in_channel_inds = [num for num,x in enumerate([int(x[-1]) for x in dig_in_pathname])
                     if x in dig_in_channel_nums]

# Check dig-in numbers and trial counts against info file
# Only if mismatch, check with user, otherwise print details and continue
# NOTE : Digital input numbers are not indices but the actual digital inputs on the board
dig_in_num_temp = [int(re.findall('[0-9]+',x)[0]) for x in dig_in_channels]
trial_count_info = info_dict['taste_params']['trial_count']
dig_in_check = sorted(dig_in_channel_nums) == sorted(dig_in_num_temp)
trial_num_check = sorted(trial_count_info) == sorted([len(x) for x in end_points])
dig_in_pathname_str = '\n'.join(dig_in_channels)

print('\n')
print('From info file' + '\n' +\
        '========================================')
print(f'Dig-ins : {dig_in_channel_nums}') 
print(f'Trial counts : {trial_count_info}') 

check_str = f'Taste dig_ins channels:\n{dig_in_pathname_str}''\n'\
        f'No. of trials: {[len(ends) for ends in end_points]}''\n'

print('\n')
print('From DAT files' + '\n' +\
        '========================================')
print(check_str)

if dig_in_check and trial_num_check:
    print('=== ALL GOOD ===')
    print('Dig-in data match with details in exp_info\n')
else:
    print('=== ALL **NOT** GOOD === \n')
    print('Dig-in data do not match with details in exp_info')
    # Show the user the number of trials on each digital input channel, 
    # and ask them to confirm to proceed
    check_bool_str, continue_bool = entry_checker(\
            msg = '\n :: Would you like to continue? (y/n) ::: ',
            check_func = lambda x: x in ['y','n'],
            fail_response = 'Please enter (y/n)')
    if continue_bool:
            if check_bool_str == 'y':
                check = True
            else:
                check = False
    else:
        print(':: Exiting ::')
        exit()

# TODO: Have separate params file for EMG
# For now, use duration params from spike-sorting
# Write out durations to the params (json) file 
#params_file = hdf5_name.split('.')[0] + ".params"
params_file_name = glob.glob('./**.params')[0]
with open(params_file_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
durations = params_dict['spike_array_durations']
print(f'Using durations ::: {dict(zip(["pre","post"], durations))}' + '\n')

# Grab the names of the arrays containing emg recordings
emg_nodes = hf5.list_nodes('/raw_emg')
emg_pathname = []
for node in emg_nodes:
    emg_pathname.append(node._v_pathname)

# Create a numpy array to store emg data by trials
# Shape : Channels x Tastes x Trials x Time
# Use max number of trials to define array, this allows people with uneven
# numbers of trials to continue working
trial_counts = [len(x) for x in start_points]
if len(np.unique(trial_counts)) > 1:
    print(f'!! Uneven numbers of trials !! {trial_counts}')
    print(f'Using {np.max(trial_counts)} as trial count')
    print('== EMG ARRAY WILL HAVE EMPTY TRIALS ==')

emg_data = np.ndarray((len(emg_pathname), 
    len(dig_in_channels), 
    np.max(trial_counts), 
    durations[0]+durations[1]))

# And pull out emg data into this array
for i in range(len(emg_pathname)):
    data = hf5.get_node(emg_pathname[i])[:]
    #exec("data = hf5.root.raw_emg.%s[:]" % emg_pathname[i].split('/')[-1])
    for j in range(len(dig_in_channels)):
        for k in range(len(start_points[dig_in_channel_nums[j]])):
            raw_emg_data = data[start_points[dig_in_channel_nums[j]][k]\
                    -durations[0]*30:start_points[dig_in_channel_nums[j]][k]\
                    +durations[1]*30]
            raw_emg_data = 0.195*(raw_emg_data)
            # Downsample the raw data by averaging the 30 samples per millisecond, 
            # and assign to emg_data
            emg_data[i, j, k, :] = np.mean(raw_emg_data.reshape((-1, 30)), axis = 1)

# Write out booleans for non-zero trials
nonzero_trial = np.abs(emg_data.mean(axis=(0,3))) > 0

# Save output in emg dir
if not os.path.exists('emg_output'):
    os.makedirs('emg_output')

# Save the emg_data
np.save('emg_output/emg_data.npy', emg_data)
np.save('emg_output/nonzero_trials.npy', nonzero_trial)

hf5.close()
