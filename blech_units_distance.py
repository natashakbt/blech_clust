# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from numba import jit

# Numba compiled function to compute the number of spikes in this_unit_times that are within 1 ms of a spike in other_unit_times, and vice versa
@jit(nogil = True)
def unit_distance(this_unit_times, other_unit_times):
	this_unit_counter = 0
	other_unit_counter = 0
	for i in range(len(this_unit_times)):
		for j in range(len(other_unit_times)):
			if np.abs(this_unit_times[i] - other_unit_times[j]) <= 1.0:
				this_unit_counter += 1
				other_unit_counter += 1
	return this_unit_counter, other_unit_counter

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
print "=================="
print "Distance calculation starting"
unit_distances = np.zeros((len(units), len(units)))
for this_unit in range(len(units)):
	this_unit_times = (units[this_unit].times[:])/30.0
	for other_unit in range(len(units)): 
		if other_unit < this_unit:
			continue
		other_unit_times = (units[other_unit].times[:])/30.0
		this_unit_counter, other_unit_counter = unit_distance(this_unit_times, other_unit_times)
		unit_distances[this_unit, other_unit] = 100.0*(float(this_unit_counter)/len(this_unit_times))
		unit_distances[other_unit, this_unit] = 100.0*(float(other_unit_counter)/len(other_unit_times))
	# Print the progress to the window
	print "Unit %i of %i completed" % (this_unit+1, len(units))
print "Distance calculation complete, results being saved to file"
print "=================="


'''
Tried to do everything in the numba compiled loop through vectorized numpy code - failed because of memory errors involved in broadcasting arrays
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
		unit_distances[other_unit, this_unit] = 100.0*len(np.where(diff_other_this <= 1.0)[0])/len(diff_other_this)'''
	
# Make a node for storing unit distances under /sorted_units. First try to delete it, and pass if it exists
try:
	hf5.remove_node('/unit_distances')
except:
	pass
hf5.create_array('/', 'unit_distances', unit_distances)

hf5.close()

