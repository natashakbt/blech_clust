# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt

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

# Get all the units from the hdf5 file
units = hf5.list_nodes('/sorted_units')

# Now go through the units one by one, and get the pairwise distances between them
# Distance is defined as the percentage of spikes of the reference unit that have a spike from the compared unit within 1 ms
unit_distances = np.zeros((len(units), len(units)))
for this_unit in range(len(units)):
	# print this_unit, this_unit_times.shape
	for other_unit in range(len(units)): 
		if other_unit < this_unit:
			continue
		this_unit_times = units[this_unit].times[:]
		other_unit_times = units[other_unit].times[:]
		# print other_unit, other_unit_times.shape
		# Tiling doesn't work in this case because it copies data, memory error happens
		# this_unit_times = np.vstack((this_unit_times for i in range(other_unit_times.shape[0])))
		# other_unit_times = np.vstack((other_unit_times for i in range(this_unit_times.shape[1])))
		# this_unit_times = np.tile(this_unit_times, (other_unit_times.shape[0], 1))
		# other_unit_times = np.tile(other_unit_times, (this_unit_times.shape[1], 1))
		# Broadcast this_unit_times to the shape of #other_unit_times x #this_unit_times. Then broadcast other_unit_times to #this_unit_times X #other_unit_times
		x_this_unit = np.zeros(len(other_unit_times))
		x_other_unit = np.zeros(len(this_unit_times))
		this_unit_times = np.broadcast_arrays(x_this_unit[:, None], this_unit_times[None, :])
		this_unit_times = this_unit_times[1]
		other_unit_times = np.broadcast_arrays(x_other_unit[:, None], other_unit_times[None, :])
		other_unit_times = other_unit_times[1]
		other_unit_times = other_unit_times.T
		diff = np.abs(this_unit_times - other_unit_times)/30.0
		del this_unit_times, other_unit_times
		diff_this_other = np.min(diff, axis = 0)
		diff_other_this = np.min(diff, axis = 1)
		unit_distances[this_unit, other_unit] = 100.0*len(np.where(diff_this_other <= 1.0)[0])/len(diff_this_other)
		unit_distances[other_unit, this_unit] = 100.0*len(np.where(diff_other_this <= 1.0)[0])/len(diff_other_this)

	# del this_unit_times, other_unit_times, diff

# Make a node for storing unit distances under /sorted_units. First try to delete it, and pass if it exists
try:
	hf5.remove_node('/sorted_units/unit_distances')
except:
	pass
hf5.create_array('/sorted_units', 'unit_distances', unit_distances)

hf5.close()

