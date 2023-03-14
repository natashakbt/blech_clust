# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from numba import jit
import json
import glob
from utils.blech_utils import imp_metadata

# Numba compiled function to compute the number of spikes in this_unit_times 
# that are within 1 ms of a spike in other_unit_times, and vice versa
@jit(nogil = True)
def unit_similarity(this_unit_times, other_unit_times):
    this_unit_counter = 0
    other_unit_counter = 0
    for this_time in this_unit_times:
        for other_time in other_unit_times:
            if abs(this_time - other_time) <= 1.0:
                    this_unit_counter += 1
                    other_unit_counter += 1
    return this_unit_counter, other_unit_counter

############################################################
############################################################

# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

params_dict = metadata_handler.params_dict
similarity_cutoff = params_dict['similarity_cutoff']

# Open a file to write these unit distance violations to - 
# these units are likely the same and one of them will need to be removed from the HDF5 file
unit_similarity_violations = open("unit_similarity_violations.txt", "w")
print("Unit number 1" + "\t" + "Unit number 2", file = unit_similarity_violations)

# Get all the units from the hdf5 file
units = hf5.list_nodes('/sorted_units')

# Now go through the units one by one, and get the pairwise distances between them
# Similarity is defined as the percentage of spikes of the reference unit 
# that have a spike from the compared unit within 1 ms
print("==================")
print("Similarity calculation starting")
print(f"Similarity cutoff ::: {similarity_cutoff}")
unit_distances = np.zeros((len(units), len(units)))
for this_unit in range(len(units)):
    this_unit_times = (units[this_unit].times[:])/30.0
    for other_unit in range(len(units)): 
        if other_unit < this_unit:
                continue
        other_unit_times = (units[other_unit].times[:])/30.0
        this_unit_counter, other_unit_counter = \
                unit_similarity(this_unit_times, other_unit_times)
        unit_distances[this_unit, other_unit] = \
                100.0*(float(this_unit_counter)/len(this_unit_times))
        unit_distances[other_unit, this_unit] = \
                100.0*(float(other_unit_counter)/len(other_unit_times))
        # If the similarity goes beyond the defined cutoff, 
        # write these unit numbers to file
        if this_unit != other_unit \
                and (unit_distances[this_unit, other_unit] > similarity_cutoff \
                or unit_distances[other_unit, this_unit] > similarity_cutoff):
                print(str(this_unit) + "\t" + \
                        str(other_unit), file = unit_similarity_violations)
    # Print the progress to the window
    print("Unit %i of %i completed" % (this_unit+1, len(units)))
print("Similarity calculation complete, results being saved to file")
print("==================")


# Make a node for storing unit distances under /sorted_units. First try to delete it, and pass if it exists
try:
	hf5.remove_node('/unit_distances')
except:
	pass
hf5.create_array('/', 'unit_distances', unit_distances)

hf5.close()
unit_similarity_violations.close()
