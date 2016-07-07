# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import LeavePOut

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

# Get the digital inputs/tastes available, then ask the user to rank them in order of palatability
trains_dig_in = hf5.list_nodes('/spike_trains')
palatability_rank = easygui.multenterbox(msg = 'Rank the digital inputs in order of palatability (1 for the lowest, only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(palatability_rank)):
	palatability_rank[i] = int(palatability_rank[i])

# Now ask the user to put in the identities of the digital inputs
identities = easygui.multenterbox(msg = 'Put in the identities of the digital inputs (only integers)', fields = [train._v_name for train in trains_dig_in])
for i in range(len(identities)):
	identities[i] = int(identities[i])

# Get the palatability/identity calculation paramaters from the user
params = easygui.multenterbox(msg = 'Enter the parameters for palatability/identity calculation', fields = ['Window size (ms)', 'Step size (ms)'])
for i in range(len(params)):
	params[i] = int(params[i])

# Get the pre-stimulus time from the user
pre_stim = easygui.multenterbox(msg = 'Enter the pre-stimulus time for the spike trains', fields = ['Pre stim (ms)'])
pre_stim = int(pre_stim[0])

# Ask the user about the type of units they want to do the calculations on (single or all units)
unit_type = easygui.multchoicebox(msg = 'Which type of units do you want to use?', choices = ('All units', 'Single units', 'Multi units'))
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
single_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 1])
multi_units = np.array([i for i in range(len(all_units)) if hf5.root.unit_descriptor[i]["single_unit"] == 0])
chosen_units = []
if unit_type[0] == 'All units':
	chosen_units = all_units
elif unit_type[0] == 'Single units':
	chosen_units = single_units
else:
	chosen_units = multi_units

# Now make arrays to pull the data out
num_trials = trains_dig_in[0].spike_array.shape[0]
num_units = len(chosen_units)
time = trains_dig_in[0].spike_array.shape[2]
num_tastes = len(trains_dig_in)
palatability = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = int)
identity = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = int)
response = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = np.dtype('float64'))
laser = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials, 2), dtype = float)

# Fill in the palatabilities and identities
for i in range(num_tastes):
	palatability[:, :, num_trials*i:num_trials*(i+1)] = palatability_rank[i]*np.ones((palatability.shape[0], palatability.shape[1], num_trials))
	identity[:, :, num_trials*i:num_trials*(i+1)] = identities[i]*np.ones((palatability.shape[0], palatability.shape[1], num_trials))

# Now fill in the responses and laser (duration,lag) tuples
for i in range(0, time - params[0] + params[1], params[1]):
	for j in range(num_units):
		for k in range(num_tastes):
			# If the lasers were used, get the appropriate durations and lags. Else assign zeros to both
			try:
				laser[i/params[1], j, num_trials*k:num_trials*(k+1)] = np.vstack((trains_dig_in[k].laser_durations[:], trains_dig_in[k].laser_onset_lag[:])).T
			except:
				laser[i/params[1], j, num_trials*k:num_trials*(k+1)] = np.zeros((num_trials, 2))
			response[i/params[1], j, num_trials*k:num_trials*(k+1)] = np.mean(trains_dig_in[k].spike_array[:, chosen_units[j], i:i + params[0]], axis = 1)

# Create an ancillary_analysis group in the hdf5 file, and write these arrays to that group
try:
	hf5.remove_node('/ancillary_analysis', recursive = True)
except:
	pass
hf5.create_group('/', 'ancillary_analysis')
hf5.create_array('/ancillary_analysis', 'palatability', palatability)
hf5.create_array('/ancillary_analysis', 'identity', identity)
hf5.create_array('/ancillary_analysis', 'laser', laser)
hf5.create_array('/ancillary_analysis', 'neural_response', response)
hf5.create_array('/ancillary_analysis', 'params', params)
hf5.create_array('/ancillary_analysis', 'pre_stim', np.array(pre_stim))
hf5.flush()

# First pull out the unique laser(duration,lag) combinations - these are the same irrespective of the unit and time
unique_lasers = np.vstack({tuple(row) for row in laser[0, 0, :, :]})
# Now get the sets of trials with these unique duration and lag combinations
trials = []
for i in range(len(unique_lasers)):
	this_trials = [j for j in range(laser.shape[2]) if np.array_equal(laser[0, 0, j, :], unique_lasers[i, :])]
	trials.append(this_trials)

# Save the trials and unique laser combos to the hdf5 file as well
hf5.create_array('/ancillary_analysis', 'trials', np.array(trials))
hf5.create_array('/ancillary_analysis', 'laser_combination_d_l', unique_lasers)
hf5.flush()

# --------Palatability calculation - separate r and p values for Spearman and Pearson correlations----------------
# Set up arrays to store palatability calculation results
r_spearman = np.zeros((unique_lasers.shape[0], palatability.shape[0], palatability.shape[1]))
p_spearman = np.ones(r_spearman.shape)
r_pearson = np.zeros(r_spearman.shape)
p_pearson = np.ones(r_spearman.shape)

for i in range(unique_lasers.shape[0]):
	for j in range(palatability.shape[0]):
		for k in range(palatability.shape[1]):
			ranks = rankdata(response[j, k, trials[i]])
			r_spearman[i, j, k], p_spearman[i, j, k] = spearmanr(ranks, palatability[j, k, trials[i]])
			r_pearson[i, j, k], p_pearson[i, j, k] = pearsonr(response[j, k, trials[i]], palatability[j, k, trials[i]])
			# Account for NaNs - happens when all spike counts are equal (esp 0)
			if np.isnan(r_spearman[i, j, k]):
				r_spearman[i, j, k] = 0.0
				p_spearman[i, j, k] = 1.0
			if np.isnan(r_pearson[i, j, k]):
				r_pearson[i, j, k] = 0.0
				p_pearson[i, j, k] = 1.0

# Move to linear discriminant analysis
lda_palatability = np.zeros((unique_lasers.shape[0], identity.shape[0]))
for i in range(unique_lasers.shape[0]):
	for j in range(identity.shape[0]):
		X = response[j, :, trials[i]] 
		Y = palatability[j, 0, trials[i]]
		# Use k-fold cross validation where k = 1 sample left out
		test_results = []
		c_validator = LeavePOut(len(Y), 1)
		for train, test in c_validator:
			model = LDA()
			model.fit(X[train, :], Y[train])
			# And test on the left out kth trial - compare to the actual class of the kth trial and store in test results
			test_results.append(np.mean(model.predict(X[test]) == Y[test]))
		lda_palatability[i, j] = np.mean(test_results)

# Save these arrays to file
hf5.create_array('/ancillary_analysis', 'r_pearson', r_pearson)
hf5.create_array('/ancillary_analysis', 'p_pearson', p_pearson)
hf5.create_array('/ancillary_analysis', 'r_spearman', r_spearman)
hf5.create_array('/ancillary_analysis', 'p_spearman', p_spearman)
hf5.create_array('/ancillary_analysis', 'lda_palatability', lda_palatability)
hf5.flush()

# --------End palatability calculation----------------------------------------------------------------------------

#---------Identity calculation - one way ANOVA between responses to the unique tastes and linear discriminant analysis (LDA)-----------------------------
f_identity = np.zeros((unique_lasers.shape[0], identity.shape[0], identity.shape[1]))
p_identity = np.ones(f_identity.shape)

for i in range(unique_lasers.shape[0]):
	for j in range(identity.shape[0]):
		for k in range(identity.shape[1]):
			# Get the unique tastes
			unique_tastes = np.unique(identity[j, k, trials[i]])
			samples = []
			# Now run through the unique tastes
			for taste in unique_tastes:
				samples.append([trial for trial in trials[i] if identity[j, k, trial] == taste])
			# Now run the one way ANOVA
			f_identity[i, j, k], p_identity[i, j, k] = f_oneway(*[response[j, k, sample] for sample in samples])
			# Again some sanity check error correction - remove NaNs, they arise when all spike counts are 0s
			if np.isnan(f_identity[i, j, k]):
				f_identity[i, j, k] = 0.0
				p_identity[i, j, k] = 1.0

# Move to linear discriminant analysis
lda_identity = np.zeros((unique_lasers.shape[0], identity.shape[0]))
for i in range(unique_lasers.shape[0]):
	for j in range(identity.shape[0]):
		X = response[j, :, trials[i]] 
		Y = identity[j, 0, trials[i]]
		# Use k-fold cross validation where k = 1 sample left out
		test_results = []
		c_validator = LeavePOut(len(Y), 1)
		for train, test in c_validator:
			model = LDA()
			model.fit(X[train, :], Y[train])
			# And test on the left out kth trial - compare to the actual class of the kth trial and store in test results
			test_results.append(np.mean(model.predict(X[test]) == Y[test]))
		lda_identity[i, j] = np.mean(test_results)
 
# Save these arrays to file
hf5.create_array('/ancillary_analysis', 'f_identity', f_identity)
hf5.create_array('/ancillary_analysis', 'p_identity', p_identity)
hf5.create_array('/ancillary_analysis', 'lda_identity', lda_identity)
hf5.flush()		

#---------End identity calculation--------------------------------------------------------------------------------

hf5.close()
