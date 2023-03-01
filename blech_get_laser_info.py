# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import json
import glob
import itertools
import pandas as pd
from tqdm import tqdm
from utils.clustering import *
from utils.blech_utils import (
        imp_metadata,
        )
from blech_process import calc_recording_cutoff

from blech_units_make_arrays import (
        get_dig_in_data,
        convert_where_to_list,
        create_laser_params_for_digin
        )

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv[1])
dir_name = metadata_handler.dir_name
os.chdir(dir_name)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
sampling_rate = params_dict['sampling_rate']
sampling_rate_ms = sampling_rate/1000

# Grab the names of the arrays containing digital inputs, 
# and pull the data into a numpy array
dig_in_pathname, dig_in_basename, dig_in_data = get_dig_in_data(hf5)
dig_in_diff = np.diff(dig_in_data,axis=-1)
# Calculate start and end points of pulses
start_points = convert_where_to_list(np.where(dig_in_diff == 1))
end_points = convert_where_to_list(np.where(dig_in_diff == -1))

# Pull out taste dig-ins
taste_digin_inds = info_dict['taste_params']['dig_ins']
taste_digin_channels = [dig_in_basename[x] for x in taste_digin_inds]
taste_str = "\n".join(taste_digin_channels)

# Extract laser dig-in from params file
laser_digin_inds = [info_dict['laser_params']['dig_in']][0]

# Pull laser digin from hdf5 file
if len(laser_digin_inds) == 0:
    laser_digin_channels = []
    laser_str = 'None'
else:
    laser_digin_channels = [dig_in_basename[x] for x in laser_digin_inds]
    laser_str = "\n".join(laser_digin_channels)

print(f'Taste dig_ins ::: \n{taste_str}\n')
print(f'Laser dig_in ::: \n{laser_str}\n')

# NOTE: Calculate headstage falling off same way for all not "none" channels 
# Pull out raw_electrode and raw_emg data
if '/raw' in hf5:
    raw_electrodes = [x for x in hf5.get_node('/','raw')]
if '/raw_emg' in hf5:
    raw_emg_electrodes = [x for x in hf5.get_node('/','raw_emg')]

all_electrodes = [raw_electrodes, raw_emg_electrodes] 
all_electrodes = [x for y in all_electrodes for x in y]
all_electrode_names = [x._v_pathname for x in all_electrodes]
electrode_names = list(zip(*[x.split('/')[1:] for x in all_electrode_names]))

cutoff_data = []
for this_el in tqdm(all_electrodes): 
    raw_el = this_el[:]
    # High bandpass filter the raw electrode recordings
    filt_el = get_filtered_electrode(
        raw_el,
        freq=[params_dict['bandpass_lower_cutoff'],
              params_dict['bandpass_upper_cutoff']],
        sampling_rate=params_dict['sampling_rate'])

    # Delete raw electrode recording from memory
    del raw_el

    this_out = calc_recording_cutoff(
                    filt_el,
                    params_dict['sampling_rate'],
                    params_dict['voltage_cutoff'],
                    params_dict['max_breach_rate'],
                    params_dict['max_secs_above_cutoff'],
                    params_dict['max_mean_breach_rate_persec']
                    ) 
    cutoff_data.append(this_out)


cutoff_frame = pd.DataFrame(
        data = cutoff_data,
        columns = [
            'breach_rate', 
            'breaches_per_sec', 
            'secs_above_cutoff', 
            'mean_breach_rate_persec',
            'recording_cutoff'
            ],
        )
cutoff_frame['electrode_type'] = all_electrode_names[0]
cutoff_frame['electrode_name'] = all_electrode_names[1]

expt_end_time = cutoff_frame['recording_cutoff'].min()*sampling_rate

# Check start points prior to loop and print results
dig_in_trials = np.array([len(x) for x in start_points])
start_points_cutoff = [x[x<expt_end_time] for x in start_points]
end_points_cutoff = [x[x<expt_end_time] for x in end_points]
trials_before_cutoff = np.array([len(x) for x in start_points_cutoff])
cutoff_frame = pd.DataFrame(
        data = dict(
            dig_ins = dig_in_basename,
            trials_before_cutoff = trials_before_cutoff,
            trials_after_cutoff = dig_in_trials - trials_before_cutoff
            )
        )
print()
print(cutoff_frame)

taste_starts_cutoff = [start_points_cutoff[i] for i in taste_digin_inds]

print()
for i, this_dig_in in enumerate(taste_starts_cutoff): 
    print(f'Creating laser info for {dig_in_basename[i]}')
    create_laser_params_for_digin(
            i,
            this_dig_in,
            start_points_cutoff,
            end_points_cutoff,
            sampling_rate,
            sampling_rate_ms,
            laser_digin_inds,
            dig_in_basename,
            hf5,
            )

#============================================================
# Correct for laser sampling errors before moving on to next step
#============================================================
# Take "correct" values from info file
# NOTE: This will likely need to be corrected later on as info file
#       currently only allows single values for onset and duration
info_laser_params = info_dict['laser_params']
info_laser_onset = info_laser_params['onset'] 
info_laser_duration = info_laser_params['duration']
info_laser_data = [(0,0), (info_laser_duration, info_laser_onset)]
info_laser_data = [np.array(x) for x in info_laser_data]

dig_in_list = hf5.get_node('/','spike_trains')
dig_in_list = [x for x in dig_in_list if 'dig_in' in x._v_pathname]
durations = [x.laser_durations[:] for x in dig_in_list]
lags = [x.laser_onset_lag[:] for x in dig_in_list]

# Compare actual laser data to calculate onsets and durations
# And correct as needed
data_tuples = [np.vstack([x,y]).T for x,y in zip(durations, lags)] 

corrected_tuples = []
for this_dig_in in data_tuples:
    deviations = np.stack(
            [np.linalg.norm(this_dig_in - x,axis=-1) for x in info_laser_data])
    min_ind = np.argmin(deviations, axis=0)
    corrected_tuples.append(np.stack([info_laser_data[x] for x in min_ind]))

for this_dig_in, this_corrected_dat in zip(dig_in_list, corrected_tuples):
    this_dig_in.laser_durations[:] = this_corrected_dat[:,0] 
    this_dig_in.laser_onset_lag[:] = this_corrected_dat[:,1] 
hf5.flush()

orig_unique_tuples = set([tuple(x) for x in np.concatenate(data_tuples)])
fin_unique_tuples = set([tuple(x) for x in np.concatenate(corrected_tuples)])

print()
print("Laser timings corrected")
print("============================================================")
print("Original data")
for x in orig_unique_tuples:
    print(x)
print("")
print("Corrected data")
for x in fin_unique_tuples:
    print(x)

#============================================================
#============================================================
# Create an ancillary_analysis group in the hdf5 file, 
# and write these arrays to that group
if '/ancillary_analysis' in hf5:
    hf5.remove_node('/ancillary_analysis', recursive = True)
hf5.create_group('/', 'ancillary_analysis')

# First pull out the unique laser(duration,lag) combinations - 
# these are the same irrespective of the unit and time
#unique_lasers = np.vstack({tuple(row) for row in laser[0, 0, :, :]})
#unique_lasers = unique_lasers[unique_lasers[:, 0].argsort(), :]
#unique_lasers = unique_lasers[unique_lasers[:, 1].argsort(), :]
unique_lasers = np.array(list(fin_unique_tuples))

# Now get the sets of trials with these unique duration and lag combinations
concat_laser_tuples = np.concatenate(corrected_tuples)
trials = np.stack(
        [
            [
                i for i, dat in enumerate(concat_laser_tuples) \
                        if np.array_equal(dat,this_cond)
                ]
            for this_cond in unique_lasers
            ]
        )

#trials = []
#for i in range(len(unique_lasers)):
#    this_trials = [j for j in range(laser.shape[2]) if np.array_equal(laser[0, 0, j, :], unique_lasers[i, :])]
#    trials.append(this_trials)
#trials = np.array(trials)

# Save the trials and unique laser combos to the hdf5 file as well
hf5.create_array('/ancillary_analysis', 'trials', trials)
hf5.create_array('/ancillary_analysis', 'laser_combination_d_l', unique_lasers)
hf5.flush()
hf5.close()
