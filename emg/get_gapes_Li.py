# Use the results in Li et al. 2016 to get gapes on taste trials

import tables
import numpy as np
import easygui
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from detect_peaks import *

# Load the gape algorithm results from Li et al. 2016
a = loadmat('QDA_nostd_no_first.mat')
a = a['important_coefficients'][0]

# Define a function that applies the QD algorithm to each individual movement, with x = interval and y = duration. Returns True or False based on if the QD evaluates to <0 or not
def QDA(x, y):
	return (a[0] + a[1]*x + a[2]*y + a[3]*x**2 + a[4]*x*y + a[5]*y**2) < 0

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

# Grab the nodes for the available tastes
trains_dig_in = hf5.list_nodes('/spike_trains')
num_trials = trains_dig_in[0].spike_array.shape[0]
num_tastes = len(trains_dig_in)

# Load the unique laser duration/lag combos and the trials that correspond to them from the ancillary analysis node
trials = hf5.root.ancillary_analysis.trials[:]
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Ask the user for the pre-stimulus time used
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', fields = ['Pre stim (ms)'])
pre_stim = int(pre_stim[0])

# Load the required emg data (the envelope and sig_trials)
env = np.load('env.npy')
sig_trials = np.load('sig_trials.npy')

# Stack up env and sig_trials
env = np.vstack(tuple(env[i, :, :] for i in range(env.shape[0])))
sig_trials = np.reshape(sig_trials, (sig_trials.shape[0]*sig_trials.shape[1]))

# Now arrange these arrays by laser condition X taste X time
env_final = np.empty((len(trials), num_tastes, int(num_trials/len(trials)), env.shape[1]), dtype = float)
sig_trials_final = np.empty((len(trials), num_tastes, int(num_trials/len(trials))), dtype = int)

# Also make an array to store the time of first gape on every trial
first_gape = np.empty((len(trials), num_tastes, int(num_trials/len(trials))), dtype = int)

# Fill up these arrays
for i in range(len(trials)):
	for j in range(num_tastes):
		env_final[i, j, :, :] = env[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)], :]
		sig_trials_final[i, j, :] = sig_trials[trials[i][np.where((trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)) == True)]]

# Make an array to store gapes (with 1s)
gapes_Li = np.zeros(env_final.shape) 

# Ask the user for the post stimulus time to consider the results upto
post_stim = easygui.multenterbox(msg = 'Enter the post-stimulus time to be used', fields = ['Post stim (ms)'])
post_stim = int(post_stim[0])

# Run through the trials and get burst times, intervals and durations. Also check if these bursts are gapes - if they are, put 1s in the gape array
for i in range(sig_trials_final.shape[0]):
	for j in range(sig_trials_final.shape[1]):
		for k in range(sig_trials_final.shape[2]):
			# Get features only if its a trial with significant EMG activity
			if sig_trials_final[i, j, k] == 1:
				# Get peak indices
				peak_ind = detect_peaks(env_final[i, j, k, :], mpd = 85, mph = np.mean(env_final[i, :, :, :pre_stim]) + np.std(env_final[i, :, :, :pre_stim]))

				# Get the indices, in the smoothed signal, that are below the mean of the smoothed signal
				below_mean_ind = np.where(env_final[i, j, k, :] <= np.mean(env_final[i, :, :, :pre_stim]))[0]

				# Throw out peaks if they happen in the pre-stim period
				accept_peaks = np.where(peak_ind > pre_stim)[0]
				peak_ind = peak_ind[accept_peaks]

				# Run through the accepted peaks, and append their breadths to durations. There might be peaks too close to the end of the trial - skip those. Append the surviving peaks to final_peak_ind
				durations = []
				final_peak_ind = []
				for peak in peak_ind:
					try:
						left_end = np.where(below_mean_ind < peak)[0][-1]
						right_end = np.where(below_mean_ind > peak)[0][0]
					except:
						continue
					dur = below_mean_ind[right_end]-below_mean_ind[left_end]
					if dur > 20.0 and dur <= 200.0:
						durations.append(dur)
						final_peak_ind.append(peak)
				durations = np.array(durations)
				peak_ind = np.array(final_peak_ind)
				
				# In case there aren't any peaks or just one peak (very unlikely), skip this trial and mark it 0 on sig_trials
				if len(peak_ind) <= 1:
					sig_trials_final[i, j, k] = 0
					continue

				# Get inter-burst-intervals for the accepted peaks, convert to Hz (from ms)
				intervals = []
				for peak in range(len(peak_ind)):
					# For the first peak, the interval is counted from the second peak
					if peak == 0:
						intervals.append(1000.0/(peak_ind[peak+1] - peak_ind[peak]))
					# For the last peak, the interval is counted from the second to last peak
					elif peak == len(peak_ind) - 1:
						intervals.append(1000.0/(peak_ind[peak] - peak_ind[peak-1]))
					# For every other peak, take the largest interval
					else:
						intervals.append(1000.0/(np.amax([(peak_ind[peak] - peak_ind[peak-1]), (peak_ind[peak+1] - peak_ind[peak])])))
				intervals = np.array(intervals)	

				# Now run through the intervals and durations of the accepted movements, and see if they are gapes. If yes, mark them appropriately in gapes_Li
				# Do not use the first movement/peak in the trial - that is usually not a gape
				for peak in range(len(durations) - 1):
					gape = QDA(intervals[peak+1], durations[peak+1])
					if gape and peak_ind[peak+1] - pre_stim <= post_stim:
						gapes_Li[i, j, k, peak_ind[peak+1]] = 1.0

				# If there are no gapes on a trial, mark these as 0 on sig_trials_final and 0 on first_gape. Else put the time of the first gape in first_gape
				if np.sum(gapes_Li[i, j, k, :]) == 0.0:
					sig_trials_final[i, j, k] = 0
					first_gape[i, j, k] = 0
				else:
					first_gape[i, j, k] = np.where(gapes_Li[i, j, k, :] > 0.0)[0][0]

# Save these results to the hdf5 file
try:
	hf5.remove_node('/ancillary_analysis/gapes_Li')
	hf5.remove_node('/ancillary_analysis/gape_trials_Li')
	hf5.remove_node('/ancillary_analysis/first_gape_Li')
except:
	pass
hf5.create_array('/ancillary_analysis', 'gapes_Li', gapes_Li)
hf5.create_array('/ancillary_analysis', 'gape_trials_Li', sig_trials_final)
hf5.create_array('/ancillary_analysis', 'first_gape_Li', first_gape)
hf5.flush()

hf5.close()




