import matplotlib
matplotlib.use('Agg')

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from blech_hmm import *
import pylab as plt

# Read blech.dir, and cd to that directory
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
	dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

# Pull out the NSLOTS - number of CPUs allotted
n_cpu = int(os.getenv('NSLOTS'))

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-10:] == 'hmm_params':
		params_file = files

# Read the .hmm_params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Assign the params to variables
min_states = int(params[0])
max_states = int(params[1])
threshold = float(params[2])
seeds = int(params[3])
edge_inertia = float(params[4])
dist_inertia = float(params[5])
taste = int(params[6])
pre_stim = int(params[7])
bin_size = int(params[8])
pre_stim_hmm = int(params[9])
post_stim_hmm = int(params[10])

# Open up hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get the spike array from the required taste/input
exec('spikes = hf5.root.spike_trains.dig_in_%i.spike_array[:]' % taste)

# Slice out the required portion of the spike array, and bin it
spikes = spikes[:, :, pre_stim - pre_stim_hmm:pre_stim + post_stim_hmm]
binned_spikes = np.zeros((spikes.shape[0], (pre_stim_hmm + post_stim_hmm)/bin_size))
time = []
for i in range(spikes.shape[0]):
	time = []
	for k in range(0, spikes.shape[2], bin_size):
		time.append(k - pre_stim_hmm)
		n_firing_units = np.where(np.sum(spikes[i, :, k:k+bin_size], axis = 1) > 0)[0]
		if n_firing_units.size:
			n_firing_units = n_firing_units + 1 
		else:
			n_firing_units = [0]
		binned_spikes[i, k/bin_size] = np.random.choice(n_firing_units) 

# Implement a Multinomial HMM for no. of states defined by min_states and max_states - run all the trials through the HMM
hmm_results = []
off_trials = np.arange(binned_spikes.shape[0])
for n_states in range(min_states, max_states + 1):
	# Run the Multinomial HMM - skip it if it doesn't converge
	try:
		result = multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia)
		hmm_results.append((n_states, result))
	except:
		continue

# Delete the multinomial_hmm_results node under /spike_trains/dig_in_(taste)/ if it exists
try:
	hf5.remove_node('/spike_trains/dig_in_%i/multinomial_hmm_results' % taste, recursive = True)
except:
	pass

# Then create the multinomial_hmm_results group
hf5.create_group('/spike_trains/dig_in_%i/' % taste, 'multinomial_hmm_results')

hf5.flush()

# Delete the Multinomial folder within HMM_plots if it exists for this taste
try:
	os.system("rm -r ./HMM_plots/dig_in_%i/Multinomial" % taste)
except:
	pass	

# Make a folder for plots of Multinomial HMM analysis
os.mkdir("HMM_plots/dig_in_%i/Multinomial" % taste)

# Go through the HMM results, and make plots for each state and each trial
for result in hmm_results:
	
	# Make a directory for this number of states
	os.mkdir("HMM_plots/dig_in_%i/Multinomial/states_%i" % (taste, result[0]))
		
	# Make a group under multinomial_hmm_results for this number of states
	hf5.create_group('/spike_trains/dig_in_%i/multinomial_hmm_results' % taste, 'states_%i' % (result[0])) 
	# Write the emission and transition probabilties to this group
	emission_labels = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'emission_labels', np.array(result[1][4][0].keys()))
	emission_matrix = []
	for i in range(len(result[1][4])):
		emission_matrix.append(result[1][4][i].values())
	emission_probs = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'emission_probs', np.array(emission_matrix))
	transition_probs = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'transition_probs', result[1][5])
	posterior_proba = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'posterior_proba', result[1][6])

	# Also write the json model string to file
	model_json = hf5.create_array('/spike_trains/dig_in_%i/poisson_hmm_results/laser/states_%i' % (taste, result[0]), 'model_json', result[1][0])

	# Write the log-likelihood, AIC/BIC score, and time vector to the hdf5 file too
	log_prob = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'log_likelihood', np.array(result[1][1]))
	aic = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'aic', np.array(result[1][2]))
	bic = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'bic', np.array(result[1][3]))
	time_vect = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/states_%i' % (taste, result[0]), 'time', time)
	hf5.flush()

	# Go through the trials in binned_spikes and plot the trial-wise posterior probabilities
	for i in range(binned_spikes.shape[0]):
		fig = plt.figure()
		for j in range(posterior_proba.shape[2]):
			plt.plot(time, posterior_proba[i, :, j])
		plt.xlabel('Time post stimulus (ms)')
		plt.ylabel('Probability of HMM states')
		plt.title('Trial %i' % (i+1))
		fig.savefig('HMM_plots/dig_in_%i/Multinomial/states_%i/Trial_%i.png' % (taste, result[0], (i+1)))
		plt.close("all")

# Check if the laser_array exists - if it does, perform a 2nd round of HMM training on just the non-laser trials
exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
laser_exists = []
try:
	laser_exists = dig_in.laser_durations[:]
except:
	pass
if len(laser_exists) > 0:
	on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
	off_trials = np.where(dig_in.laser_array[:] == 0.0)[0]
	# Implement a Multinomial HMM for no. of states defined by min_states and max_states
	hmm_results = []
	for n_states in range(min_states, max_states + 1):
		# Run Multinomial HMM - skip if it doesn't converge
		try:
			result = multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia)
			hmm_results.append((n_states, result))
		except:
			continue

	# Delete the laser node under /spike_trains/dig_in_(taste)/multinomial_hmm_results/ if it exists
	try:
		exec("hf5.remove_node('/spike_trains/dig_in_%i/multinomial_hmm_results/laser' % taste, recursive = True)")
	except:
		pass

	# Then create the multinomial_hmm_results group
	exec("hf5.create_group('/spike_trains/dig_in_%i/multinomial_hmm_results' % taste, 'laser')")
	hf5.flush()

	# Delete the laser folder within HMM_plots/Multinomial if it exists for this taste
	try:
		os.system("rm -r ./HMM_plots/dig_in_%i/Multinomial/laser" % taste)
	except:
		pass	

	# Make a folder for plots of Multinomial HMM analysis
	os.mkdir("HMM_plots/dig_in_%i/Multinomial/laser" % taste)
	
	# Go through the HMM results, and make plots for each state and each trial
	for result in hmm_results:
		
		# Make a directory for this number of states
		os.mkdir("HMM_plots/dig_in_%i/Multinomial/laser/states_%i" % (taste, result[0]))
		
		# Make a group under multinomial_hmm_results for this number of states
		hf5.create_group('/spike_trains/dig_in_%i/multinomial_hmm_results/laser' % taste, 'states_%i' % (result[0])) 
		# Write the emission and transition probabilties to this group
		emission_labels = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'emission_labels', np.array(result[1][4][0].keys()))
		emission_matrix = []
		for i in range(len(result[1][4])):
			emission_matrix.append(result[1][4][i].values())
		emission_probs = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'emission_probs', np.array(emission_matrix))
		transition_probs = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'transition_probs', result[1][5])
		posterior_proba = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'posterior_proba', result[1][6])

		# Also write the json model string to file
		model_json = hf5.create_array('/spike_trains/dig_in_%i/poisson_hmm_results/laser/states_%i' % (taste, result[0]), 'model_json', result[1][0])

		# Write the log-likelihood and AIC/BIC score to the hdf5 file too
		log_prob = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'log_likelihood', np.array(result[1][1]))
		aic = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'aic', np.array(result[1][2]))
		bic = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'bic', np.array(result[1][3]))
		time_vect = hf5.create_array('/spike_trains/dig_in_%i/multinomial_hmm_results/laser/states_%i' % (taste, result[0]), 'time', time)
		hf5.flush()

		# Go through the trials in binned_spikes and plot the trial-wise posterior probabilities
		for i in range(binned_spikes.shape[0]):
			if i in on_trials:
				label = 'laser_on_'
			else:
				label = 'laser_off_'
			fig = plt.figure()
			for j in range(posterior_proba.shape[2]):
				plt.plot(time, posterior_proba[i, :, j])
			plt.xlabel('Time post stimulus (ms)')
			plt.ylabel('Probability of HMM states')
			plt.title('Trial %i' % (i+1))
			fig.savefig('HMM_plots/dig_in_%i/Multinomial/laser/states_%i/%sTrial_%i.png' % (taste, result[0], label, (i+1)))
			plt.close("all")
		
hf5.close()


