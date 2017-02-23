# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import pymc3 as pm

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

# Get the digital inputs/tastes available
trains_dig_in = hf5.list_nodes('/spike_trains')

# Get the pre-stimulus time from the user
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', fields = ['Pre stim (ms)'])
pre_stim = int(pre_stim[0])

# Ask the user about the type of units they want to do the calculations on (single or all units)
unit_type = easygui.multchoicebox(msg = 'Which type of units do you want to use?', choices = ('All units', 'Single units', 'Multi units', 'Custom choice'))
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
multi_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 0])
chosen_units = []
if unit_type[0] == 'All units':
	chosen_units = all_units
elif unit_type[0] == 'Single units':
	chosen_units = single_units
elif unit_type[0] == 'Multi units':
	chosen_units = multi_units
else:
	chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose?', choices = ([i for i in all_units]))
	for i in range(len(chosen_units)):
		chosen_units[i] = int(chosen_units[i])
	chosen_units = np.array(chosen_units)

# Get the laser duration/lag combos
lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
# Now sort the durations/lags in ascending order - the first combo is the control/laser off condition (0, 0) after sorting
lasers = lasers[lasers[:, 0].argsort(), :]
lasers = lasers[lasers[:, 1].argsort(), :]

# Ask the user for the number of MCMC samples wanted for estimation of firing rates
num_samples = easygui.multenterbox(msg = 'Enter the number of MCMC samples you want to estimate firing rates (2000 is reasonable)', fields = ['Number of MCMC samples'])
num_samples = int(num_samples[0])

# Make an array to store the results of the laser effect analysis of dimensions units X laser_conditions X tastes X MCMC samples X 2 (laser on/off)
results = np.zeros((len(chosen_units), len(lasers) - 1, len(trains_dig_in), num_samples, 2))
# Also make an array to store the Gelman-Rubin convergence statistic values
#gelman_rubin_rhat = np.empty()

# Run through the chosen units
# Include a progress counter
print("====================================")
print("Starting Bayesian analysis of the effects of laser on firing rate of units")
for unit in range(len(chosen_units)):
	# And run through the laser conditions: (except the first one, aka control)
	for laser_status in range(lasers.shape[0] - 1):
		condition = lasers[laser_status + 1, :]
		print("Running: Unit: {} of {}, Laser duration: {}, Lag: {}".format(unit + 1, len(chosen_units), condition[0], condition[1]))
		# And finally, run through the tastes and pull all the data into respective arrays
		spikes = []
		tastes = []
		laser_condition = []		
		for stimulus in range(len(trains_dig_in)):
			# Get the correct trial numbers for this laser condition
			trials = np.where((trains_dig_in[stimulus].laser_durations[:] == condition[0])*(trains_dig_in[stimulus].laser_onset_lag[:] == condition[1]))[0]
			# Also get the control trials
			controls = np.where((trains_dig_in[stimulus].laser_durations[:] == 0.0)*(trains_dig_in[stimulus].laser_onset_lag[:] == 0.0))[0]

			# Append the data from these trials to spikes
			spikes.append(np.sum(trains_dig_in[stimulus].spike_array[trials, chosen_units[unit], pre_stim + condition[1]:pre_stim + condition[1] + condition[0]], axis = 1))
			spikes.append(np.sum(trains_dig_in[stimulus].spike_array[controls, chosen_units[unit], pre_stim + condition[1]:pre_stim + condition[1] + condition[0]], axis = 1))
			
			# Append the condition markers (0 for laser off, 1 for laser on)
			laser_condition.append([1 for i in range(len(trials))])
			laser_condition.append([0 for i in range(len(controls))])

			# Append the taste markers
			tastes.append([stimulus for i in range(len(trials) + len(controls))])

		# Convert all the data to numpy arrays
		spikes = np.array(spikes).flatten()
		tastes = np.array(tastes).flatten()
		laser_condition = np.array(laser_condition).flatten()

		# Make pymc3 model for these data - assume: 1.) Poisson emissions, 2.) log link. Estimate main effects for tastes and laser conditions, plus interactions
		with pm.Model() as model:
			a = pm.Normal('a', mu = 0, sd = 1000)
			b_t = pm.Normal('b_t', mu = 0, sd = 10, shape = 4)
			b_l = pm.Normal('b_l', mu = 0, sd = 10, shape = 2)
			b_t_l = pm.Normal('b_t_l', mu = 0, sd = 10, shape = (4, 2))

			p = np.exp(a + b_t[tastes] + b_l[laser_condition] + b_t_l[tastes, laser_condition])

			output = pm.Poisson('spikes', mu = p, observed = spikes)

		# Sample from this model
		with model:
			trace = pm.sample(num_samples, init = 'advi_map')

		# The strategy now is to run through the MCMC samples, and calculate the difference in the Poisson mean between the laser on and off conditions for the different tastes
		# Run through the tastes again
		for stimulus in range(len(trains_dig_in)):
			# First calculate the mean firing rate for the laser off (control) condition for this taste
			results[unit, laser_status, stimulus, :, 0] = np.exp(trace['a'] + trace['b_t'][:, stimulus] + trace['b_l'][:, 0] + trace['b_t_l'][:, stimulus, 0])

			# Then calculate the mean firing rate for the laser on condition for this taste
			results[unit, laser_status, stimulus, :, 1] = np.exp(trace['a'] + trace['b_t'][:, stimulus] + trace['b_l'][:, 1] + trace['b_t_l'][:, stimulus, 1])

print("Bayesian analysis of the effects of laser on firing rate of units finished")
print("====================================")

# Store the results in the hdf5 file in /root/laser_effects_bayesian
# Check if the node already exists, delete it if it does, and remake it
try:
	hf5.remove_node('/laser_effects_bayesian', recursive = True)
except:
	pass
hf5.create_group('/', 'laser_effects_bayesian')
# Then save the data to the file
hf5.create_array('/laser_effects_bayesian', 'laser_combination_d_l', lasers)
hf5.create_array('/laser_effects_bayesian', 'mean_firing_rates', results)

hf5.close()






			
			
				 

		
			 


			
			
