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

# Get the list of spike trains by digital input channels
trains_dig_in = hf5.list_nodes('/spike_trains')

# Taste responsiveness calculation parameters
r_pre_stim = 500
r_post_stim = 2500

# Plot PSTHs by digital input channels
for dig_in in trains_dig_in:
	os.mkdir('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1])
	trial_avg_spike_array = np.mean(dig_in.spike_array[:], axis = 0)
	for unit in range(trial_avg_spike_array.shape[0]):
		time = []
		spike_rate = []
		for i in range(0, trial_avg_spike_array.shape[1] - params[0], params[1]):
			time.append(i - pre_stim)
			spike_rate.append(1000.0*np.sum(trial_avg_spike_array[unit, i:i+params[0]])/float(params[0]))
		taste_responsiveness_t, taste_responsiveness_p = ttest_ind(np.mean(dig_in.spike_array[:, unit, pre_stim:pre_stim + r_post_stim], axis = 1), np.mean(dig_in.spike_array[:, unit, pre_stim - r_pre_stim:pre_stim], axis = 1))   
		fig = plt.figure()
		plt.title('Unit: %i, Window size: %i ms, Step size: %i ms, Taste responsive: %s' % (unit + 1, params[0], params[1], str(bool(taste_responsiveness_p<0.001))))
		plt.xlabel('Time from taste delivery (ms)')
		plt.ylabel('Firing rate (Hz)')
		plt.plot(time, spike_rate, linewidth = 3.0)
		fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i.png' % (unit + 1))
		plt.close("all")
		
		# Check if the laser_array exists, and plot laser PSTH if it does
		if dig_in.laser_array:
			on_trials = np.where(dig_in.laser_array[:] == 1.0)[0]
			off_trials = np.where(dig_in.laser_array[:] == 0.0)[0]
			trial_avg_on_array = np.mean(dig_in.spike_array[on_trials, :, :], axis = 0)
			trial_avg_off_array = np.mean(dig_in.spike_array[off_trials, :, :], axis = 0)
			time = []
			on_spike_rate = []
			off_spike_rate = []
			for i in range(0, trial_avg_spike_array.shape[1] - params[0], params[1]):
				time.append(i - pre_stim)
				on_spike_rate.append(1000.0*np.sum(trial_avg_on_array[unit, i:i+params[0]])/float(params[0]))						
				off_spike_rate.append(1000.0*np.sum(trial_avg_off_array[unit, i:i+params[0]])/float(params[0]))
			fig = plt.figure()
			plt.title('Unit: %i laser PSTH, Window size: %i ms, Step size: %i ms' % (unit + 1, params[0], params[1]))
			plt.xlabel('Time from taste delivery (ms)')
			plt.ylabel('Firing rate (Hz)')
			plt.plot(time, on_spike_rate, linewidth = 3.0, color = 'g')
			plt.plot(time, off_spike_rate, linewidth = 3.0, color = 'k')			
			fig.savefig('./PSTH/'+str.split(dig_in._v_pathname, '/')[-1]+'/Unit%i_laser_psth.png' % (unit + 1))
			plt.close("all")

hf5.close()

		
				



	


