#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:35:53 2017

@author: bradly
"""

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Grab the names of the arrays containing digital inputs, ask how many digital inputs to split file in to, and how many trials in each input
dig_in_nodes = hf5.list_nodes('/digital_in')
add_node_number = easygui.integerbox(msg='You have ' +str(len(dig_in_nodes)) + ' digital input channels, how many will you add?', default='4',lowerbound=0,upperbound=10)
trial_split = easygui.multenterbox(msg = 'Put in the number of trials to parse from each of the original input channels (only integers)', fields = [node._v_name for node in dig_in_nodes[:add_node_number]], values = ['15' for node in dig_in_nodes[:add_node_number]])

#Convert all values to integers
trial_split = list(map(int,trial_split))

# Grab array information for each digital input channel, split into first and last sections, place in corresponding digitial input group array
for node in range(len(dig_in_nodes)):
    exec("full_array = hf5.root.spike_trains.dig_in_%i.spike_array[:] " % node)
    hf5.create_group('/spike_trains', str.split('dig_in_%s' % str(node+add_node_number), '/')[-1])
    hf5.create_array('/spike_trains/dig_in_%s' % str(node+add_node_number), 'spike_array', np.array(full_array[trial_split[node]:full_array.shape[0],:,:]))
    
    # Revove oringal dig_in group, recreate it, and populate with first array
    hf5.remove_node('/spike_trains/dig_in_%s' % str(node), recursive = True)
    hf5.create_group('/spike_trains', str.split('dig_in_%s' % str(node), '/')[-1])
    hf5.create_array('/spike_trains/dig_in_%s' % str(node), 'spike_array', np.array(full_array[0:trial_split[node],:,:]))

spike_array_nodes_after = hf5.list_nodes('/spike_trains')

# Ask if they want to delete the spare trials (the arrays created to account for uneven trials)
msg   = "Do you want to delete the 'short' spike arrays (i.e. the leftover arrays from making equal trials per tastant)?"
array_delete = easygui.buttonbox(msg,choices = ["Yes","No"])
if array_delete == "Yes":
    for array in range(len(trial_split)):
        	#Delete data
        	hf5.remove_node('/spike_trains/dig_in_%s' % str(array+add_node_number), recursive = True)

hf5.flush()
hf5.close()    
 
