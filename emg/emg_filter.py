# Subtracts the two emg signals and filters and saves the results.

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, freqz
import easygui
import os
import sys
from tqdm import tqdm
import shutil
import json
import glob
import pandas as pd

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

os.chdir(dir_name)

# Load the data
# shape : channels x tastes x trials x time
emg_data = np.load('emg_output/emg_data.npy')

# Extract info experimental info file
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)

## Ask the user for stimulus delivery time in each trial, and convert to an integer
## todo : Can be pulled from info file
#pre_stim = easygui.multenterbox(
#        msg = 'Enter the pre-stimulus time included in each trial', 
#        fields = ['Pre-stimulus time (ms)']) 
#pre_stim = int(pre_stim[0])

# Pull pre_stim duration from params file
params_file_name = glob.glob('./**.params')[0]
with open(params_file_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
durations = params_dict['spike_array_durations']
pre_stim = int(durations[0])
print(f'Using pre-stim duration : {pre_stim}' + '\n')

# Get coefficients for Butterworth filters
m, n = butter(2, 2.0*300.0/1000.0, 'highpass')
c, d = butter(2, 2.0*15.0/1000.0, 'lowpass')

## todo: This can be pulled from info file
## check how many EMG channels used in this experiment
#check = easygui.ynbox(
#        msg = 'Did you have more than one EMG channel?', 
#        title = 'Check YES if you did')
layout_path = glob.glob(os.path.join(dir_name,"*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path) 

# Allow for multiple emg CAR groups
#wanted_electrodes = info_dict['emg']['electrodes']
#select_bool = [x in wanted_electrodes for x in electrode_layout_frame.electrode_num] 
wanted_rows = pd.DataFrame(
        [x for num,x in electrode_layout_frame.iterrows() \
                if 'emg' in x.CAR_group])
wanted_rows = wanted_rows.sort_values('electrode_ind')
wanted_rows.reset_index(inplace=True, drop=True)

#wanted_rows = electrode_layout_frame.loc[select_bool] 
#wanted_rows = electrode_layout_frame.loc[] 
print('Using electrodes :')
print(wanted_rows)
print()
#check = len(info_dict['emg']['electrodes']) > 1

# TODO: Ask about differencing pairs
# Difference by CAR emg labels
# If only 1 channel per emg CAR, do not difference if asked
emg_car_groups = [x[1] for x in wanted_rows.groupby('CAR_group')]
emg_car_names = [x.CAR_group.unique()[0] for x in emg_car_groups]
emg_car_inds = [x.index.values for x in emg_car_groups]

print('EMG CAR Groups with more than 1 channel will be differenced')
print('EMG CAR groups as follows:')
for x in emg_car_groups:
    print(x)
    print()

# TODO: This question can go into an EMG params file
# todo: Rename to diff_data at this stage
# Bandpass filter the emg signals, and store them in a numpy array. 
# Low pass filter the bandpassed signals, and store them in another array
# Take difference between pairs of channels
# Shape : Channels x Tastes x Trials x Time
emg_data_grouped = [emg_data[x] for x in emg_car_inds]
emg_diff_data = []
for x in emg_data_grouped:
    if len(x) > 1:
        emg_diff_data.append(np.squeeze(np.diff(x,axis=0)))
    elif len(x) > 2:
        raise Exception("More than 2 per EMG CAR currently not supported")
    else:
        emg_diff_data.append(np.squeeze(x))

#if len(emg_data) > 2:
#    print('More than 2 emg channels...currently not working\n')
#    emg_list = np.split(emg_data,2,axis=0)
#    emg_diff_data = np.squeeze(np.stack([np.diff(x,axis=0) for x in emg_list]))
#elif len(emg_data) == 1:
#    print('Only 1 emg channel...no differencing\n')
#    # If only single channel
#    emg_diff_data = emg_data[np.newaxis]
#else:
#    # If only one pair of channels
#    # Keep as 4D array so shape is consistent
#    print('2 emg emg_channels...differencing\n')
#    emg_diff_data = np.diff(emg_data, axis=0) 

# Iterate over trials and apply frequency filter
iters = list(np.ndindex(emg_diff_data[0].shape[:-1])) 
emg_filt_list = []
emg_env_list = []
for x in emg_diff_data:
    emg_filt = np.zeros(x.shape)
    emg_env = np.zeros(x.shape)
    for this_iter in iters:
        temp_filt = filtfilt(m, n, x[this_iter[0], this_iter[1]])
        emg_filt[this_iter[0], this_iter[1]] = temp_filt 
        emg_env[this_iter[0], this_iter[1]] = filtfilt(c, d, np.abs(temp_filt))
    emg_filt_list.append(emg_filt)
    emg_env_list.append(emg_env)

#for i in range(emg_diff_data.shape[1]):
#    for j in range(emg_diff_data.shape[2]):
#        if check:
#            emg_filt[i, j, :] = \
#                    filtfilt(m, n, emg_diff_data[0, i, j, :] - emg_diff_data[1, i, j, :])
#        else:
#            emg_filt[i, j, :] = filtfilt(m, n, emg_diff_data[0, i, j, :])
#        env[i, j, :] = filtfilt(c, d, np.abs(emg_filt[i, j, :]))    

sig_trials_list = []
for i in range(len(emg_diff_data)):
    ## Get mean and std of baseline emg activity, 
    ## and use it to select trials that have significant post stimulus activity
    # sig_trials (assumed shape) : tastes x trials
    pre_m = np.mean(np.abs(emg_filt_list[i][...,:pre_stim]), axis = (2))
    pre_s = np.std(np.abs(emg_filt_list[i][...,:pre_stim]), axis = (2))

    post_m = np.mean(np.abs(emg_filt_list[i][...,pre_stim:]), axis = (2))
    post_max = np.max(np.abs(emg_filt_list[i][...,pre_stim:]), axis = (2))

    # If any of the channels passes the criteria, select that trial as significant
    # 1) mean post-stim activity > mean pre-stim activity
    # 2) max post-stim activity > mean pre-stim activity + 4*pre-stim STD
    #mean_bool = np.sum(post_m > pre_m, axis = 0) > 0
    #std_bool = np.sum(post_max > (pre_m + 4.0*pre_s), axis = 0) > 0
    mean_bool = post_m > pre_m
    std_bool = post_max > (pre_m + 4.0*pre_s)

    # Logical AND
    sig_trials = mean_bool * std_bool
    sig_trials_list.append(sig_trials)

#m = np.mean(np.abs(emg_filt[:, :, :pre_stim]))
#s = np.std(np.abs(emg_filt[:, :, :pre_stim]))
#for i in range(emg_diff_data.shape[1]):
#    for j in range(emg_diff_data.shape[2]):
#        if np.mean(np.abs(emg_filt[i, j, pre_stim:])) > m \
#                and np.max(np.abs(emg_filt[i, j, pre_stim:])) > m + 4.0*s:
#            sig_trials[i, j] = 1

# NOTE: Currently DIFFERENT sig_trials for each channel 
# Save the highpass filtered signal, 
# the envelope and the indicator of significant trials as a np array
# Iterate over channels and save them in different directories 
for num,this_name in enumerate(emg_car_names): 
    #dir_path = f'emg_output/emg_channel{num}'
    dir_path = f'emg_output/{this_name}'
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    # emg_filt (output shape): tastes x trials x time
    np.save(os.path.join(dir_path, f'emg_filt.npy'), emg_filt_list[num])
    # env (output shape): tastes x trials x time
    np.save(os.path.join(dir_path, f'emg_env.npy'), emg_env_list[num])
    # sig_trials (output shape): tastes x trials
    np.save(os.path.join(dir_path, 'sig_trials.npy'), sig_trials_list[num])
