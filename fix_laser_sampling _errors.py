#
# Since the digital inputs are being sampled at 30kHz, sometimes the laser durations (or onsets)
# are recorded for a few ms more or less than the intended length of the pulse. 
# We scale the length of the pulse by factors of 10 - so 
# sometimes a 2500ms pulse can become 2510 or 2490ms. This gives errors in later steps.


import tables
import numpy as np
import easygui
import os

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Get the names of all files in the current directory, and find the hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open up the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

trains_dig_in = hf5.list_nodes('/spike_trains')

# Ask the user for the correct laser duration and convert to integers
num_durations = easygui.multenterbox(msg = "How many laser durations were used in this experiment?", fields = ["Number of Laser durations"])
num_durations = int(num_durations[0])

durations = easygui.multenterbox(msg = "What are the laser durations used in this experiment (control duration [i.e., 0 ms] added automatically)?", fields = ["Duration{}".format(num + 1) for num in range(num_durations)])
for i in range(len(durations)):
    durations[i] = int(durations[i])
durations.append(0)

 

# Ask for the correct laser onset latencies
num_latencies = easygui.multenterbox(msg = "How many latencies were used in this experiment?", fields = ["Number of Laser onset latencies"])
num_latencies = int(num_latencies[0])

latencies = easygui.multenterbox(msg = "What are the latencies of laser onset used for this experiment?", fields = ["Latency{}".format(num + 1) for num in range(num_latencies)])
for i in range(len(latencies)):
    latencies[i] = int(latencies[i])
latencies.append(0)

    

# Checking the laser-duration array to find sampling errors and correct them
for dig_in in range(len(trains_dig_in)):
    for duration in range(len(trains_dig_in[dig_in].laser_durations)):
        if trains_dig_in[dig_in].laser_durations[duration] not in durations:
            diff = np.absolute(np.array(durations) - trains_dig_in[dig_in].laser_durations[duration])
            trains_dig_in[dig_in].laser_durations[duration] = durations[np.argmin(diff)]
hf5.flush()

# Checking the laser onset latency array to find sampling errors and correct them
for dig_in in range(len(trains_dig_in)):
    for latency in range(len(trains_dig_in[dig_in].laser_onset_lag)):
        if trains_dig_in[dig_in].laser_onset_lag[latency] not in latencies:
            diff = np.absolute(np.array(latencies) - trains_dig_in[dig_in].laser_onset_lag[latency])
            trains_dig_in[dig_in].laser_onset_lag[latency] = latencies[np.argmin(diff)]
hf5.flush()


hf5.close()

