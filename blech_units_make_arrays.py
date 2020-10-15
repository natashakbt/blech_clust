# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import re
import json
import glob

def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ',exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

# Ask for the directory where the hdf5 file sits, and change to that directory
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
	exec("dig_in.append(hf5.root.digital_in.%s[:])" % dig_in_pathname[-1].split('/')[-1])
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
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.json'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)

dig_in_channel_nums = info_dict['taste_params']['dig_ins']
dig_in_channels = [dig_in_pathname[i] for i in dig_in_channel_nums]
dig_in_channel_inds = np.arange(len(dig_in_channels))

# Check dig-in numbers and trial counts against info file
# Only if mismatch, check with user, otherwise print details and continue
# NOTE : Digital input numbers are not indices but the actual digital inputs on the board
dig_in_num_temp = [int(re.findall('[0-9]+',x)[0]) for x in dig_in_channels]
trial_count_info = info_dict['taste_params']['trial_count']
dig_in_check = sorted(dig_in_channel_nums) == sorted(dig_in_num_temp)
trial_num_check = sorted(trial_count_info) == sorted([len(x) for x in end_points])
dig_in_pathname_str = '\n'.join(dig_in_channels)
check_str = f'Taste dig_ins channels:\n{dig_in_pathname_str}''\n'\
        f'No. of trials: {[len(ends) for ends in end_points]}''\n'
print(check_str)
if dig_in_check and trial_num_check:
    print('=== ALL GOOD ===')
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
    
# Ask the user which digital input channels should be used for getting 
# spike train data, and convert the channel numbers into integers for pulling 
# stuff out of change_points
#def cluster_check(x):
#    clusters = re.findall('[0-9]+',x)
#    return sum([i.isdigit() for i in clusters]) == len(clusters)
#chosen_msg, continue_bool = entry_checker(\
#        msg = f'Please select digins from {dig_in_pathname} (anything separated) '\
#        '\n:: "111" for all ::\n',
#        check_func = cluster_check,
#        fail_response = 'Please enter integers')
#if continue_bool:
#    chosen_digin = re.findall('[0-9]+|-[0-9]+',chosen_msg)
#    dig_in_channel_inds = [int(x) for x in chosen_digin]
#    # If 111, select all
#    if 111 in dig_in_channel_inds:
#        dig_in_channel_inds = np.arange(len(dig_in_pathname))    
#    dig_in_channels = [dig_in_pathname[i] for i in dig_in_channel_inds]
#    print(f'Chosen dig_ins {dig_in_channels}'\
#            '\n=====================\n')
#else:
#    print(':: Exiting ::')
#    exit()

#dig_in_channels = easygui.multchoicebox(\
#        msg = 'Which digital input channels should be used to '\
#                'produce spike train data trial-wise?', 
#        choices = ([path for path in dig_in_pathname]))

#dig_in_channel_inds = []
#for i in range(len(dig_in_pathname)):
#	if dig_in_pathname[i] in dig_in_channels:
#		dig_in_channel_inds.append(i)

# Extract laser dig-in from params file
laser_nums = info_dict['laser_params']['dig_in']
if len(laser_nums) == 0:
    lasers = []
    laser_str = 'None'
else:
    lasers = [dig_in_pathname[i] for i in laser_nums]
    laser_str = "\n".join(lasers)

taste_str = "\n".join(dig_in_channels)

#print(f'Taste dig_ins ::: \n{taste_str}\n')
print(f'Laser dig_in ::: \n{laser_str}\n')

# Ask the user which digital input channels should be used for conditioning 
# the stimuli channels above (laser channels for instance)
#chosen_msg, continue_bool = entry_checker(\
#        msg = f'Please select LASER from {dig_in_pathname} (anything separated) '\
#        '\n:: "111" for all ::\n <BLANK> for None',
#        check_func = cluster_check,
#        fail_response = 'Please enter integers')
#if continue_bool:
#    chosen_digin = re.findall('[0-9]+|-[0-9]+',chosen_msg)
#    if len(chosen_digin) == 0:
#        lasers = []
#    else:
#        laser_nums = [int(x) for x in chosen_digin]
#        # If 111, select all
#        if 111 in laser_nums:
#            laser_nums = range(len(dig_in_pathname))   
#        lasers = [dig_in_pathname[i] for i in laser_nums]
#    print(f'Chosen LASER {lasers}'\
#        '\n=====================\n')
#else:
#    print(':: Exiting ::')
#    exit()

#lasers = easygui.multchoicebox(\
#        msg = 'Which digital input channels were used for lasers? '\
#                'Click clear all and continue if you did not use lasers', 
#        choices = ([path for path in dig_in_pathname]))
laser_nums = []
if lasers:
	for i in range(len(dig_in_pathname)):
		if dig_in_pathname[i] in lasers:
			laser_nums.append(i)

# Ask the user for the pre and post stimulus durations to be pulled out, 
# and convert to integers
#durations = easygui.multenterbox(\
#        msg = 'What are the signal durations pre and post stimulus that you want to pull out', 
#        fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'])
#for i in range(len(durations)):
#	durations[i] = int(durations[i])

# Write out durations to the params (json) file 
#params_file = hdf5_name.split('.')[0] + ".params"
params_file_name = glob.glob('./**.params')[0]
with open(params_file_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
durations = params_dict['spike_array_durations']
print(f'Using durations ::: {durations}')

# Delete the spike_trains node in the hdf5 file if it exists, and then create it
try:
	hf5.remove_node('/spike_trains', recursive = True)
except:
	pass
hf5.create_group('/', 'spike_trains')

# Get list of units under the sorted_units group. 
# Find the latest/largest spike time amongst the units, 
# and get an experiment end time 
# (to account for cases where the headstage fell off mid-experiment)

units = hf5.list_nodes('/sorted_units')
expt_end_time = 0
for unit in units:
	if unit.times[-1] > expt_end_time:
		expt_end_time = unit.times[-1]

# ____                              _             
#|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
#| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                           |___/ 

# Go through the dig_in_channel_inds and make an array of spike trains 
# of dimensions (# trials x # units x trial duration (ms)) - 
# use END of digital input pulse as the time of taste delivery
for i in range(len(dig_in_channel_inds)):
    spike_train = []
    for j in range(len(start_points[dig_in_channel_inds[i]])):
        # Skip the trial if the headstage fell off before it
        if start_points[dig_in_channel_inds[i]][j] >= expt_end_time:
                continue
        # Otherwise run through the units and convert their spike times to milliseconds
        else:
                spikes = np.zeros((len(units), durations[0] + durations[1]))
                for k in range(len(units)):
                        # Get the spike times around the end of taste delivery
                        spike_times = np.where(\
                                (units[k].times[:] <= start_points[dig_in_channel_inds[i]][j]\
                                + durations[1]*30)*(units[k].times[:] >= \
                                start_points[dig_in_channel_inds[i]][j] - durations[0]*30))[0]
                        spike_times = units[k].times[spike_times]
                        spike_times = spike_times - start_points[dig_in_channel_inds[i]][j]
                        spike_times = (spike_times/30).astype(int) + durations[0]
                        # Drop any spikes that are too close to the ends of the trial
                        spike_times = spike_times[\
                                np.where((spike_times >= 0)*(spike_times < durations[0] + \
                                durations[1]))[0]]
                        spikes[k, spike_times] = 1
                        #for l in range(durations[0] + durations[1]):
                        #	spikes[k, l] = \
                        #        len(np.where((units[k].times[:] >= \
                        #        start_points[dig_in_channel_inds[i]][j] - \
                        #        (durations[0]-l)*30)*(units[k].times[:] < \
                        #        start_points[dig_in_channel_inds[i]][j] - \
                        #        (durations[0]-l-1)*30))[0])
                                
        # Append the spikes array to spike_train 
        spike_train.append(spikes)

    # And add spike_train to the hdf5 file
    hf5.create_group('/spike_trains', str.split(dig_in_channels[i], '/')[-1])
    spike_array = hf5.create_array(\
                        '/spike_trains/%s' % str.split(dig_in_channels[i], '/')[-1], \
                        'spike_array', np.array(spike_train))
    hf5.flush()

    # Make conditional stimulus array for this digital input if lasers were used
    if laser_nums:
            cond_array = np.zeros(len(end_points[dig_in_channel_inds[i]]))
            laser_start = np.zeros(len(end_points[dig_in_channel_inds[i]]))
            # Also make an array to note down the firing of the lasers one by one - 
            # for experiments where only 1 laser was fired at a time. 
            # This has 3 sorts of laser on conditions - 
            # each laser on alone, and then both on together
            laser_single = np.zeros((len(end_points[dig_in_channel_inds[i]]), 2))
            for j in range(len(end_points[dig_in_channel_inds[i]])):
                    # Skip the trial if the headstage fell off before it - 
                    # mark these trials by -1
                    if end_points[dig_in_channel_inds[i]][j] >= expt_end_time:
                            cond_array[j] = -1
                    # Else run through the lasers and check if the lasers 
                    # went off within 5 secs of the stimulus delivery time
                    for laser in range(len(laser_nums)):
                            on_trial = np.where(\
                                    np.abs(end_points[laser_nums[laser]] - \
                                    end_points[dig_in_channel_inds[i]][j]) <= 5*30000)[0]
                            if len(on_trial) > 0:
                                    # Mark this laser appropriately in the laser_single array
                                    laser_single[j, laser] = 1.0
                                    # If the lasers did go off around stimulus delivery, 
                                    # get the duration and start time in ms 
                                    # (from end of taste delivery) of the laser trial 
                                    # (as a multiple of 10 - so 53 gets rounded off to 50)
                                    cond_array[j] = 10*int((end_points[laser_nums[laser]][on_trial][0] - start_points[laser_nums[laser]][on_trial][0])/300)
                                    laser_start[j] = 10*int((start_points[laser_nums[laser]][on_trial][0] - end_points[dig_in_channel_inds[i]][j])/300)
            # Write the conditional stimulus duration array to the hdf5 file
            laser_durations = hf5.create_array('/spike_trains/%s' % str.split(dig_in_channels[i], '/')[-1], 'laser_durations', cond_array)
            laser_onset_lag = hf5.create_array('/spike_trains/%s' % str.split(dig_in_channels[i], '/')[-1], 'laser_onset_lag', laser_start)
            on_laser = hf5.create_array('/spike_trains/%s' % str.split(dig_in_channels[i], '/')[-1], 'on_laser', laser_single)
            hf5.flush() 

hf5.close()
