# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import ast
import pylab as plt
from scipy.stats import ttest_ind

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

# Ask the user for the pre stimulus duration used while making the spike arrays
pre_stim = easygui.multenterbox(msg = 'What was the pre-stimulus duration pulled into the spike arrays?', fields = ['Pre stimulus (ms)'])
pre_stim = int(pre_stim[0])

# Get the psth paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for making the PSTHs', fields = ['Window size (ms)', 'Step size (ms)'])
for i in range(len(params)):
	params[i] = int(params[i])

# Make directory to store the PSTH plots. Delete and remake the directory if it exists
try:
	os.system('rm -r '+'./PSTH')
except:
	pass
os.mkdir('./PSTH')

# Make directory to store the raster plots. Delete and remake the directory if it exists
try:
	os.system('rm -r '+'./raster')
except:
	pass
os.mkdir('./raster')

# Get the list of spike trains by digital input channels
trains_dig_in = hf5.list_nodes('/spike_trains')

# Taste responsiveness calculation parameters
r_pre_stim = 500
r_post_stim = 2500

# Plot PSTHs and rasters by digital input channels
for dig_in in trains_dig_in:
	os.mkdir('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1])
	os.mkdir('./raster/'+str.split(dig_in._v_pathname, '/')[-1])
	trial_avg_spike_array = np.mean(dig_in.spike_array[:], axis = 0)
	for unit in range(trial_avg_spike_array.shape[0]):
		time = []
		spike_rate = []
		for i in range(0, trial_avg_spike_array.shape[1] - params[0], params[1]):
			time.append(i - pre_stim)
			spike_rate.append(1000.0*np.sum(trial_avg_spike_array[unit, i:i+params[0]])/float(params[0]))
		taste_responsiveness_t, taste_responsiveness_p = ttest_ind(np.mean(dig_in.spike_array[:, unit, pre_stim:pre_stim + r_post_stim], axis = 1), np.mean(dig_in.spike_array[:, unit, pre_stim - r_pre_stim:pre_stim], axis = 1))   
		fig = plt.figure()
		plt.title('Unit: %i, Window size: %i ms, Step size: %i ms, Taste responsive: %s' % (unit + 1, params[0], params[1], str(bool(taste_responsiveness_p<0.001))) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
		plt.xlabel('Time from taste delivery (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, spike_rate, linewidth = 3.0)
		fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit + 1))
		plt.close("all")

		# Now plot the rasters for this digital input channel and unit
		# Run through the trials
		time = np.arange(dig_in.spike_array[:].shape[2] + 1) - pre_stim
		fig = plt.figure()
		for trial in range(dig_in.spike_array[:].shape[0]):
			x = np.where(dig_in.spike_array[trial, unit, :] > 0.0)[0]
			plt.vlines(x, trial, trial + 1, colors = 'black')
		plt.xticks(np.arange(0, dig_in.spike_array[:].shape[2] + 1, 500), time[::500])
		plt.yticks(np.arange(0, dig_in.spike_array[:].shape[0] + 1, 5))
		plt.title('Unit: %i raster plot' % (unit + 1) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))	
		plt.xlabel('Time from taste delivery (ms)')
		plt.ylabel('Trial number')
		fig.savefig('./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit + 1))
		plt.close("all")
		
		# Check if the laser_array exists, and plot laser PSTH if it does
		laser_exists = []		
		try:
			laser_exists = dig_in.laser_durations[:]
		except:
			pass
		if len(laser_exists) > 0:
			# First get the unique laser onset times (from end of taste delivery) in this dataset
			onset_lags = np.unique(dig_in.laser_onset_lag[:])
			# Then get the unique laser onset durations
			durations = np.unique(dig_in.laser_durations[:])

			# Then go through the combinations of the durations and onset lags and get and plot an averaged spike_rate array for each set of trials
			fig = plt.figure()
			for onset in onset_lags:
				for duration in durations:
					spike_rate = []
					time = []
					these_trials = np.where((dig_in.laser_durations[:] == duration)*(dig_in.laser_onset_lag[:] == onset) > 0)[0]
					# If no trials have this combination of onset lag and duration (can happen when duration = 0, laser off), break out of the loop
					if len(these_trials) == 0:
						continue
					trial_avg_array = np.mean(dig_in.spike_array[these_trials, :, :], axis = 0)
					for i in range(0, trial_avg_array.shape[1] - params[0], params[1]):
						time.append(i - pre_stim)
						spike_rate.append(1000.0*np.sum(trial_avg_array[unit, i:i+params[0]])/float(params[0]))
					# Now plot the PSTH for this combination of duration and onset lag
					plt.plot(time, spike_rate, linewidth = 3.0, label = 'Dur: %i ms, Lag: %i ms' % (int(duration), int(onset)))

			plt.title('Unit: %i laser PSTH, Window size: %i ms, Step size: %i ms' % (unit + 1, params[0], params[1]) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
			plt.xlabel('Time from taste delivery (ms)')
			plt.ylabel('Firing rate (Hz)')
			plt.legend(loc = 'upper left', fontsize = 10)
			fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i_laser_psth.png' % (unit + 1))
			plt.close("all")

			# And do the same to get the rasters
			for onset in onset_lags:
				for duration in durations:
					time = np.arange(dig_in.spike_array[:].shape[2] + 1) - pre_stim
					these_trials = np.where((dig_in.laser_durations[:] == duration)*(dig_in.laser_onset_lag[:] == onset) > 0)[0]
					# If no trials have this combination of onset lag and duration (can happen when duration = 0, laser off), break out of the loop
					if len(these_trials) == 0:
						continue
					fig = plt.figure()
					# Run through the trials
					for i in range(len(these_trials)):
						x = np.where(dig_in.spike_array[these_trials[i], unit, :] > 0.0)[0]
						plt.vlines(x, i, i + 1, colors = 'black')	
					plt.xticks(np.arange(0, dig_in.spike_array[:].shape[2] + 1, 500), time[::500])
					plt.yticks(np.arange(0, len(these_trials) + 1, 5))
					plt.title('Unit: %i Dur: %i ms, Lag: %i ms' % (unit + 1, int(duration), int(onset)) + '\n' + 'Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))	
					plt.xlabel('Time from taste delivery (ms)')
					plt.ylabel('Trial number')
					fig.savefig('./raster/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i_Dur%ims_Lag%ims.png' % (unit + 1, int(duration), int(onset)))
					plt.close("all")	
						
hf5.close()

		
				



	


