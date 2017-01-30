# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import itertools
from scipy.stats import rankdata
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from scipy.misc import comb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

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


# Now make arrays to pull the data out
num_trials = trains_dig_in[0].spike_array.shape[0]
num_units = len(chosen_units)
time = trains_dig_in[0].spike_array.shape[2]
num_tastes = len(trains_dig_in)
palatability = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = int)
identity = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = int)
response = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = np.dtype('float64'))
unscaled_response = np.empty(shape = ((time - params[0])/params[1] + 1, num_units, num_tastes*num_trials), dtype = np.dtype('float64'))
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
			unscaled_response[i/params[1], j, num_trials*k:num_trials*(k+1)] = np.mean(trains_dig_in[k].spike_array[:, chosen_units[j], i:i + params[0]], axis = 1)

# Now scale the responses by the maximum firing of each neuron in each trial, and save that in response
for j in range(unscaled_response.shape[1]):
	for k in range(unscaled_response.shape[2]):
		# Remember to add 1 in the denominator - the max can be 0 sometimes
		response[:, j, k] = unscaled_response[:, j, k]/(1.0 + np.max(unscaled_response[:, j, k]))
		#response[:, j, k] = unscaled_response[:, j, k]/(1.0 + np.sum(unscaled_response[:, :, k], axis = 1)) 
		#response[:, j, k] = 1000.0*unscaled_response[:, j, k]

# Create an ancillary_analysis group in the hdf5 file, and write these arrays to that group
try:
	hf5.remove_node('/ancillary_analysis', recursive = True)
except:
	pass
hf5.create_group('/', 'ancillary_analysis')
hf5.create_array('/ancillary_analysis', 'palatability', palatability)
hf5.create_array('/ancillary_analysis', 'identity', identity)
hf5.create_array('/ancillary_analysis', 'laser', laser)
hf5.create_array('/ancillary_analysis', 'scaled_neural_response', response)
hf5.create_array('/ancillary_analysis', 'unscaled_neural_response', unscaled_response)
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
trials = np.array(trials)

# Save the trials and unique laser combos to the hdf5 file as well
hf5.create_array('/ancillary_analysis', 'trials', trials)
hf5.create_array('/ancillary_analysis', 'laser_combination_d_l', unique_lasers)
hf5.flush()

#---------Taste similarity calculation (use cosine similarity)----------------------------------------------------
# Also calculate Euclidean/Mahalanobis distance between each pair of tastes in each laser condition
# Also restructure the scaled neural response array by # laser conditions X time X # tastes X # units X trials. Save this array to file as well
neural_response_laser = np.empty((unique_lasers.shape[0], (time - params[0])/params[1] + 1, num_tastes, num_units, num_trials/unique_lasers.shape[0]), dtype = np.dtype('float64'))
for i in range(unique_lasers.shape[0]):
	for j in range((time - params[0])/params[1] + 1):
		for k in range(num_tastes):
			neural_response_laser[i, j, k, :, :] = response[j, :, trials[i][np.where((trials[i] >= num_trials*k)*(trials[i] < num_trials*(k+1)) == True)[0]]].T 

# Set up an array to store similarity calculation results - similarity of every taste to every other taste at each time point in every laser condition
taste_cosine_similarity = np.empty((unique_lasers.shape[0], (time - params[0])/params[1] + 1, num_tastes, num_tastes), dtype = np.dtype('float64'))
taste_euclidean_distance = np.empty((unique_lasers.shape[0], (time - params[0])/params[1] + 1, num_tastes, num_tastes), dtype = np.dtype('float64'))
#taste_mahalanobis_distance = np.empty((unique_lasers.shape[0], (time - params[0])/params[1] + 1, num_tastes, num_tastes), dtype = np.dtype('float64'))
for i in range(unique_lasers.shape[0]):
	for j in range((time - params[0])/params[1] + 1):
		for k in range(num_tastes):
			for l in range(num_tastes):
				taste_cosine_similarity[i, j, k, l] = np.mean(cosine_similarity(neural_response_laser[i, j, k, :, :].T, neural_response_laser[i, j, l, :, :].T))
				taste_euclidean_distance[i, j, k, l] = np.mean(cdist(neural_response_laser[i, j, k, :, :].T, neural_response_laser[i, j, l, :, :].T, metric = 'euclidean'))
				# Can't run Mahalanobis distance in situations where num_units > num_trials in every laser condition. The covariance matrix cannot be inverted - so instead, reduce dimensions by running a PCA
				# Eventually not doing Mahalanobis because its not informative and causes issues with singular covariance matrices
				#pca = PCA(n_components = 3)
				#taste_mahalanobis_distance[i, j, k, l] = np.mean(cdist(pca.fit_transform(neural_response_laser[i, j, k, :, :].T), pca.fit_transform(neural_response_laser[i, j, l, :, :].T), metric = 'mahalanobis'))
				
# Save these arrays to file
hf5.create_array('/ancillary_analysis', 'taste_cosine_similarity', taste_cosine_similarity)
hf5.create_array('/ancillary_analysis', 'taste_euclidean_distance', taste_euclidean_distance)
#hf5.create_array('/ancillary_analysis', 'taste_mahalanobis_distance', taste_mahalanobis_distance)
hf5.flush()

#---------End taste similarity calculation-------------------------------------------------------------------------

#---------Taste discriminability/responsiveness calculation (ANOVA in user-defined bins)------------------------------------------
# Ask the user for the number/size of bins to use
bin_params = easygui.multenterbox(msg = 'Enter the number of bins and their size for taste discriminability/responsiveness analysis', fields = ['Number of bins (integers only)', 'Width of bins (ms)'])
for i in range(len(bin_params)):
	bin_params[i] = int(bin_params[i])

# Ask the user for the significance level to use for the taste discrimination ANOVA
discrim_p = easygui.multenterbox(msg = 'Enter the significance level to use for taste discrimination/responsiveness ANOVA', fields = ['p value'])
discrim_p = float(discrim_p[0])

# Run through the bins, and find the neurons that have significantly different firing than baseline (-2000ms to 0ms) for any of the tastes in any of these bins
responsive_neurons = []
for i in range(bin_params[0]):
	x = np.arange(0, time - params[0] + params[1], params[1]) - pre_stim
	places = np.where((x >= bin_params[1]*i)*(x <= bin_params[1]*(i+1)))[0]
	baseline_places = np.where((x >= -2000)*( x <= 0))[0]
	for j in range(num_units):
		f, p = f_oneway(np.mean(response[places, j, :], axis = 0), np.mean(response[baseline_places, j, :], axis = 0))
		# Some sanity check error correction - remove NaNs, they arise when all spike counts are 0s
		if np.isnan(f):
			f = 0.0
			p = 1.0
		# If the ANOVA gives a significant p value, append the unit number to discriminating neurons
		if p <= discrim_p and (chosen_units[j] not in responsive_neurons):
			responsive_neurons.append(chosen_units[j])
responsive_neurons = np.sort(responsive_neurons)

# Run through the bins, and find the neurons that have significantly different firing for any of the tastes in any of these bins
discriminating_neurons = []
for i in range(bin_params[0]):
	x = np.arange(0, time - params[0] + params[1], params[1]) - pre_stim
	places = np.where((x >= bin_params[1]*i)*(x < bin_params[1]*(i+1)))[0]
	for j in range(num_units):
		f, p = f_oneway(*[np.mean(response[places, j, num_trials*k:num_trials*(k+1)], axis = 0) for k in range(num_tastes)])
		# Some sanity check error correction - remove NaNs, they arise when all spike counts are 0s
		if np.isnan(f):
			f = 0.0
			p = 1.0
		# If the ANOVA gives a significant p value, append the unit number to discriminating neurons
		if p <= discrim_p and (chosen_units[j] not in discriminating_neurons):
			discriminating_neurons.append(chosen_units[j])
discriminating_neurons = np.sort(discriminating_neurons)

# Open a file to save the identities of the taste discriminating and responsive neurons
f = open('discriminative_responsive_neurons.txt', 'w')
print>>f, "Taste discriminative neurons"
for neuron in discriminating_neurons:
	print>>f, neuron
print>>f, "Taste responsive neurons"
for neuron in responsive_neurons:
	print>>f, neuron
f.close()

# Save the taste discriminating/responsive neurons to the hdf5 file	
hf5.create_array('/ancillary_analysis', 'taste_discriminating_neurons', discriminating_neurons)	
hf5.create_array('/ancillary_analysis', 'taste_responsive_neurons', responsive_neurons)
hf5.flush()	

#---------End taste discriminability calculation-------------------------------------------------------------------

#---------Taste discriminability time course (T tests between pairs of tastes)-------------------------------------

# Make an array to store the results of taste discriminability time course analysis
p_discriminability = np.empty((unique_lasers.shape[0], (time - params[0])/params[1] + 1, num_tastes, num_tastes, num_units), dtype = np.dtype('float64'))

for i in range(unique_lasers.shape[0]):
	for j in range((time - params[0])/params[1] + 1):
		for k in range(num_tastes):
			for l in range(num_tastes):
				for m in range(num_units):
					t, p = ttest_ind(neural_response_laser[i, j, k, m, :], neural_response_laser[i, j, l, m, :], equal_var = False)
					if np.isnan(p):
						p_discriminability[i, j, k, l, m] = 1.0
					else:
						p_discriminability[i, j, k, l, m] = p

# Save the taste discriminability time course to file
hf5.create_array('/ancillary_analysis', 'p_discriminability', p_discriminability)
hf5.flush()

#---------End taste discriminability time course-------------------------------------------------------------------

#---------Palatability rank order deduction-----------------------------------------------------------------------------------
# Use the mean firing of neurons in a user-defined time bin (usually 700-1200ms) and find the rank order of palatabilities that gives the highest linear/Pearson correlation

# Ask the user for the limits of the bin to use for palatability deduction
p_deduce_params = easygui.multenterbox(msg = 'Enter the start and end times to use for palatability deduction', fields = ['Start time (ms)', 'End time (ms)'])
for i in range(len(p_deduce_params)):
	p_deduce_params[i] = int(p_deduce_params[i])

# Open a file to write the results of palatability deduction
f = open('palatability_deduction.txt', 'w')

# The basic possible palatability patterns - permutations of these give all possible palatability rank orders
base_p_patterns = [[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 2], [1, 1, 2, 3], [1, 2, 2, 3], [1, 2, 3, 4]]

# Find the times/places from the neural_response_laser array that we need in the analysis
x = np.arange(0, time - params[0] + params[1], params[1]) - pre_stim
places = np.where((x >= p_deduce_params[0])*(x <= p_deduce_params[1]))[0]

# Run through the laser conditions
for i in range(unique_lasers.shape[0]):
	print>>f, "Laser condition: ", unique_lasers[i, :]
	# Run through the basic palatability patterns	
	for pattern in base_p_patterns:
		order = []
		corrs = []
		# Run through all permutations of this pattern
		for per in itertools.permutations(pattern):
    			order.append(per)
			this_corr = []
			# Run through the units
			for unit in range(num_units):
				# Get correlation for 1.) ith laser condition, 2.) in times indicated by places, 3.) for this unit, and 4.) this permutation of the basic pattern
				this_corr.append(pearsonr(np.mean(neural_response_laser[i, places, :, unit, :], axis = 0).T.reshape(-1), np.tile(per, neural_response_laser.shape[-1]))[0]**2)
			corrs.append(np.mean(this_corr))
		# Now get the order with the maximum average correlation across units
		print>>f, "Base pattern: ", pattern, " Max pattern: ", order[np.argmax(corrs)], " Max avg corr: ", np.max(corrs)
	print>>f, ""

f.close()
		
#---------End palatability rank order deduction--------------------------------------------------------------------

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
		c_validator = LeavePOut(1)
		for train, test in c_validator.split(X, Y):
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
		c_validator = LeavePOut(1)
		for train, test in c_validator.split(X, Y):
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

#---------Pairwise identity calculation---------------------------------------------------------------------------
# First pick out the unique identities in the dataset
unique_identities = np.unique(identities)
pairwise_identity = np.zeros((unique_lasers.shape[0], identity.shape[0], unique_identities.shape[0], unique_identities.shape[0]))
for i in range(unique_lasers.shape[0]):
	for j in range(identity.shape[0]):
		for k in range(unique_identities.shape[0]):
			for l in range(unique_identities.shape[0]):
				this_pair_trials = np.where((identity[j, 0, :] == unique_identities[k]) + (identity[j, 0, :] == unique_identities[l]))[0]
				X = response[j, :, [trial for trial in trials[i] if trial in this_pair_trials]]
				Y = identity[j, 0, [trial for trial in trials[i] if trial in this_pair_trials]]

				# Use k-fold cross validation k = n_splits
				test_results = []
				c_validator = StratifiedShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)
				for train, test in c_validator.split(X, Y):
					model = GaussianNB()
					model.fit(X[train, :], Y[train])
					test_results.append(model.score(X[test, :], Y[test]))
				pairwise_identity[i, j, k, l] = np.mean(test_results)

# Save this array to file
hf5.create_array('/ancillary_analysis', 'pairwise_NB_identity', pairwise_identity)		
hf5.flush()

#-----------------------------------------------------------------------------------------------------------------

hf5.close()
