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
	this_unit_times = units[this_unit].times[:]
	for other_unit in range(len(units)): 
		if other_unit < this_unit:
			continue
		other_unit_times = units[other_unit].times[:]
		# The outer keyword can be attached to any numpy ufunc to apply that operation to every element in x AND in y. So here we calculate diff[i, j] = this_unit_times[i] - other_unit_times[j] for all i and j
		diff = np.abs(np.subtract.outer(this_unit_times, other_unit_times))
		# Divide the diffs by 30 to convert to milliseconds - then check how many spikes have a spike in the other unit within 1 millisecond	
		diff_this_other = np.min(diff, axis = 1)/30.0
		diff_other_this = np.min(diff, axis = 0)/30.0
		unit_distances[this_unit, other_unit] = 100.0*len(np.where(diff_this_other <= 1.0)[0])/len(diff_this_other)
		unit_distances[other_unit, this_unit] = 100.0*len(np.where(diff_other_this <= 1.0)[0])/len(diff_other_this)
	
# Make a node for storing unit distances under /sorted_units. First try to delete it, and pass if it exists
try:
	hf5.remove_node('/sorted_units/unit_distances')
except:
	pass
hf5.create_array('/sorted_units', 'unit_distances', unit_distances)

hf5.close()

