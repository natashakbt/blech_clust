# Converts from NeuroNexus (.nex) format to the blech_clust HDF5 format. Data files recorded on the old Plexon system were saved as .plx files - after spike sorting, they were saved as .nex files. So we are assuming that these files have been sorted.
# IMPORTANT: ONLY WORKS ON PYTHON 2.7.X - NEO IS AN OLD PACKAGE THAT HAS NOT BEEN UPDATED TO PYTHON 3

# Import stuff
from neo.io import NeuroExplorerIO as nex
import numpy as np
import easygui
import tables
import os
import sys

# Check if the user is using python 2.x
assert sys.version_info < (3,0)

# Ask the user to choose the nex file they want to convert
filename = easygui.fileopenbox(msg = "Choose the .nex file you want to convert to HDF5")

# Ask the user to choose a directory to save the converted HDF5 file in
dir_name = easygui.diropenbox(msg = "Choose the directory to save the converted HDF5 file in")
# Change to that directory
os.chdir(dir_name)
# Make sub-directory with the filename
os.mkdir(str.split(str.split(filename, '/')[-1], '.nex')[0])
# Change to the sub-directory
os.chdir(str.split(str.split(filename, '/')[-1], '.nex')[0])

# Open a new HDF5 file with that filename
hf5 = tables.open_file(str.split(str.split(filename, '/')[-1], '.nex')[0] + '.h5', 'w')

# Open up the file using neo
reader = nex(filename)

# Neo structures files in segments - we mostly have only 1 segment in our Plexon recordings. Read that segment
seg = reader.read_segment()

# Get the event names in this recording (corresponds to digital inputs) - ask the user which ones to use to splice the spike trains into trials
check = easygui.ynbox(msg = 'Digital input channels: ' + str([event.annotations['channel_name'] for event in seg.events]) + '\n' + 'No. of trials: ' + str([event.times.shape for event in seg.events]), title = 'Check and confirm the number of trials detected on digital input channels')

# Go ahead only if the user approves by saying yes
if check:
	pass
else:
	print("Well, if you don't agree, blech_clust can't do much!")
	sys.exit()

# Ask the user which digital input channels should be used for getting spike train data, and convert the channel numbers into integers for pulling spikes out
dig_in_pathname = [event.annotations['channel_name'] for event in seg.events]
dig_in_channels = easygui.multchoicebox(msg = 'Which digital input channels should be used to produce spike train data trial-wise?', choices = ([event.annotations['channel_name'] for event in seg.events]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
	if dig_in_pathname[i] in dig_in_channels:
		dig_in_channel_nums.append(i)

# Ask the user for the pre and post stimulus durations to be pulled out, and convert to integers
durations = easygui.multenterbox(msg = 'What are the signal durations pre and post stimulus that you want to pull out', fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'])
for i in range(len(durations)):
	durations[i] = int(durations[i])

# Delete the spike_trains node in the hdf5 file if it exists, and then create it
try:
	hf5.remove_node('/spike_trains', recursive = True)
except:
	pass
hf5.create_group('/', 'spike_trains')

# Get the sorted units in this dataset
units = seg.spiketrains

# Make a mock unit_descriptor table too - in the old system, we weren't saving the units as RSU/FS/multi unit, but the downstream analysis code expects it. So we will make a mock table here
# Define a unit_descriptor class to be used to add things (anything!) about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
	electrode_number = tables.Int32Col()
	single_unit = tables.Int32Col()
	regular_spiking = tables.Int32Col()
	fast_spiking = tables.Int32Col()

# Go through the dig_in_channel_nums and make an array of spike trains of dimensions (# trials x # units x trial duration (ms)) - use end of digital input pulse as the time of taste delivery
# Run through the chosen digital inputs
for i in range(len(dig_in_channels)):
	spike_train = []
	# Run through the trials of each digital input
	for j in range(seg.events[dig_in_channel_nums[i]].times.shape[0]):
		# Run through the units and convert their spike times to milliseconds
		spikes = np.zeros((len(units), durations[0] + durations[1]))
		for k in range(len(units)):
			# Get the spike times around the end of taste delivery (times are stored as seconds in .nex files - converting to ms)
			spike_times = np.where((np.array(units[k].times)*1000 <= float(seg.events[dig_in_channel_nums[i]].times[j])*1000 + durations[1])*(np.array(units[k].times)*1000 >= float(seg.events[dig_in_channel_nums[i]].times[j])*1000 - durations[0]))[0]
			spike_times = 1000*np.array(units[k].times)[spike_times]
			spike_times = spike_times - float(seg.events[dig_in_channel_nums[i]].times[j])*1000
			spike_times = spike_times.astype('int') + durations[0]
			# Drop any spikes that are too close to the ends of the trial
			spike_times = spike_times[np.where((spike_times >= 0)*(spike_times < durations[0] + durations[1]))[0]]
			spikes[k, spike_times] = 1
			#for l in range(durations[0] + durations[1]):
			#	spikes[k, l] = len(np.where((units[k].times[:] >= end_points[dig_in_channel_nums[i]][j] - (durations[0]-l)*30)*(units[k].times[:] < end_points[dig_in_channel_nums[i]][j] - (durations[0]-l-1)*30))[0])
					
		# Append the spikes array to spike_train 
		spike_train.append(spikes)
	# And add spike_train to the hdf5 file
	hf5.create_group('/spike_trains', 'dig_in_%i' % i)
	spike_array = hf5.create_array('/spike_trains/dig_in_%i' % i, 'spike_array', np.array(spike_train))
	hf5.flush()

# Make a mock unit_descriptor table too - in the old system, we weren't saving the units as RSU/FS/multi unit, but the downstream analysis code expects it. So we will make a mock table here
table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
for i in range(len(units)):
	unit_description = table.row
	unit_description['electrode_number'] = 0
	unit_description['single_unit'] = 0
	unit_description['regular_spiking'] = 0
	unit_description['fast_spiking'] = 0
	unit_description.append()
	table.flush()
	hf5.flush()

hf5.close()




