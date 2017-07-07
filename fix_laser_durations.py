#How to change value in hf5 File due to Laser sampling error (high sampling rate)

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
duration = easygui.multenterbox(msg = 'What is the laser duration used for this experiment?', fields = ['Laser Duration (ms)'])
duration = int(duration[0])

# Checking the laser-duration array to find sampling errors and correct them
for i in range(len(trains_dig_in)):    
    exec('items = np.logical_and((trains_dig_in[{}].laser_durations[:] != 0.0), (trains_dig_in[{}].laser_durations[:] != duration))'.format(i,i))
    if sum(items) > 0:
        places = np.where(items)[0]
        exec('trains_dig_in[{}].laser_durations[places] = duration'.format(i))

items = np.logical_and((trains_dig_in[3].laser_durations[:] != 0.0), (trains_dig_in[0].laser_durations[:] != duration))
    if sum(items) > 0:
        places = np.where(items)[0]
        exec('trains_dig_in[{}].laser_durations[places] = duration'.format(i))



hf5.flush()

hf5.close()
