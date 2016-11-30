# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt

# Ask the user for the hdf5 files that need to be plotted together
dirs = []
while True:
	dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
	try:
		if len(dir_name) > 0:	
			dirs.append(dir_name)
	except:
		break

# Now run through the directories, and pull out the data
unique_lasers = []
r_pearson = []
r_spearman = []
p_pearson = []
p_spearman = []
p_identity = []
lda_palatability = []
lda_identity = []
taste_cosine_similarity = []
taste_euclidean_distance = []
#taste_mahalanobis_distance = []
pairwise_NB_identity = []
p_discriminability = []
pre_stim = []
params = []
num_units = 0
for dir_name in dirs:
	# Change to the directory
	os.chdir(dir_name)
	# Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	# Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r')

	# Pull the data from the /ancillary_analysis node
	unique_lasers.append(hf5.root.ancillary_analysis.laser_combination_d_l[:])
	r_pearson.append(hf5.root.ancillary_analysis.r_pearson[:])
	r_spearman.append(hf5.root.ancillary_analysis.r_spearman[:])
	p_pearson.append(hf5.root.ancillary_analysis.p_pearson[:])
	p_spearman.append(hf5.root.ancillary_analysis.p_spearman[:])
	p_identity.append(hf5.root.ancillary_analysis.p_identity[:])
	lda_palatability.append(hf5.root.ancillary_analysis.lda_palatability[:])
	lda_identity.append(hf5.root.ancillary_analysis.lda_identity[:])
	taste_cosine_similarity.append(hf5.root.ancillary_analysis.taste_cosine_similarity[:])
	taste_euclidean_distance.append(hf5.root.ancillary_analysis.taste_euclidean_distance[:])
	#taste_mahalanobis_distance.append(hf5.root.ancillary_analysis.taste_mahalanobis_distance[:])
	pairwise_NB_identity.append(hf5.root.ancillary_analysis.pairwise_NB_identity[:])
	p_discriminability.append(hf5.root.ancillary_analysis.p_discriminability[:])
	# Reading single values from the hdf5 file seems hard, needs the read() method to be called
	pre_stim.append(hf5.root.ancillary_analysis.pre_stim.read())
	params.append(hf5.root.ancillary_analysis.params[:])
	# Also maintain a counter of the number of units in the analysis
	num_units += hf5.root.ancillary_analysis.palatability.shape[1]

	# Close the hdf5 file
	hf5.close()

# Check if the number of laser activation/inactivation windows is same across files, raise an error and quit if it isn't
if all(unique_lasers[i].shape == unique_lasers[0].shape for i in range(len(unique_lasers))):
	pass
else:
	print "Number of inactivation/activation windows doesn't seem to be the same across days. Please check and try again"
	sys.exit()

# Now first set the ordering of laser trials straight across data files
laser_order = []
for i in range(len(unique_lasers)):
	# The first file defines the order	
	if i == 0:
		laser_order.append(np.arange(unique_lasers[i].shape[0]))
	# And everyone else follows
	else:
		this_order = []
		for j in range(unique_lasers[i].shape[0]):
			for k in range(unique_lasers[i].shape[0]):
				if np.array_equal(unique_lasers[0][j, :], unique_lasers[i][k, :]):
					this_order.append(k)
		laser_order.append(np.array(this_order))

# Now join up all the data into big numpy arrays, maintaining the laser order defined in laser_order
# If there's only one data file, set the final arrays to the only array read in
if len(laser_order) == 1:
	r_pearson = r_pearson[0]
	r_spearman = r_spearman[0]
	p_pearson = p_pearson[0]
	p_spearman = p_spearman[0]
	p_identity = p_identity[0]
	lda_palatability = lda_palatability[0]
	lda_identity = lda_identity[0]
	taste_cosine_similarity = taste_cosine_similarity[0]
	taste_euclidean_distance = taste_euclidean_distance[0]
	#taste_mahalanobis_distance = taste_mahalanobis_distance[0]
	pairwise_NB_identity = pairwise_NB_identity[0]
	p_discriminability = p_discriminability[0]
else:
	r_pearson = np.concatenate(tuple(r_pearson[i][laser_order[i], :, :] for i in range(len(r_pearson))), axis = 2)
	r_spearman = np.concatenate(tuple(r_spearman[i][laser_order[i], :, :] for i in range(len(r_spearman))), axis = 2)
	p_pearson = np.concatenate(tuple(p_pearson[i][laser_order[i], :, :] for i in range(len(p_pearson))), axis = 2)
	p_spearman = np.concatenate(tuple(p_spearman[i][laser_order[i], :, :] for i in range(len(p_spearman))), axis = 2)
	p_identity = np.concatenate(tuple(p_identity[i][laser_order[i], :, :] for i in range(len(p_identity))), axis = 2)
	lda_palatability = np.stack(tuple(lda_palatability[i][laser_order[i], :] for i in range(len(lda_palatability))), axis = -1)
	lda_identity = np.stack(tuple(lda_identity[i][laser_order[i], :] for i in range(len(lda_identity))), axis = -1)
	taste_cosine_similarity = np.stack(tuple(taste_cosine_similarity[i][laser_order[i], :] for i in range(len(taste_cosine_similarity))), axis = -1)
	taste_euclidean_distance = np.stack(tuple(taste_euclidean_distance[i][laser_order[i], :] for i in range(len(taste_euclidean_distance))), axis = -1)
	#taste_mahalanobis_distance = np.stack(tuple(taste_mahalanobis_distance[i][laser_order[i], :] for i in range(len(taste_mahalanobis_distance))), axis = -1)
	pairwise_NB_identity = np.stack(tuple(pairwise_NB_identity[i][laser_order[i], :, :, :] for i in range(len(pairwise_NB_identity))), axis = -1)
	p_discriminability = np.concatenate(tuple(p_discriminability[i][laser_order[i], :, :] for i in range(len(p_discriminability))), axis = 4)
	# Now average the lda and distance results along the last axis (i.e across sessions)
	lda_palatability = np.mean(lda_palatability, axis = 2)
	lda_identity = np.mean(lda_identity, axis = 2)
	taste_cosine_similarity = np.mean(taste_cosine_similarity, axis = -1)
	taste_euclidean_distance = np.mean(taste_euclidean_distance, axis = -1)
	pairwise_NB_identity = np.mean(pairwise_NB_identity, axis = -1)
	#taste_mahalanobis_distance = np.mean(taste_mahalanobis_distance, axis = -1)

# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for palatability/identity analysis')
os.chdir(dir_name)

# Ask the user for the significance level and # of significant windows to use for plotting the identity/palatability p-values
p_values = easygui.multenterbox(msg = 'Enter the significance criteria for palatability/identity calculation', fields = ['Significance level (p value)', 'Number of consecutive significant windows'])
p_values[0] = float(p_values[0])
p_values[1] = int(p_values[1])

# Get the x array for all the plotting
x = np.arange(0, r_pearson.shape[1]*params[0][1], params[0][1]) - pre_stim[0]

# Ask the user for the time limits to be plotted
time_limits = easygui.multenterbox(msg = 'Enter the time limits for plotting the results', fields = ['Start time (ms)', 'End time (ms)']) 
for i in range(len(time_limits)):
	time_limits[i] = int(time_limits[i])
plot_indices = np.where((x>=time_limits[0])*(x<=time_limits[1]))[0]

# Save all these arrays in the output directory
np.save('r_pearson.npy', r_pearson)
np.save('r_spearman.npy', r_spearman)
np.save('p_pearson.npy', p_pearson)
np.save('p_spearman.npy', p_spearman)
np.save('p_identity.npy', p_identity)
np.save('lda_palatability.npy', lda_palatability)
np.save('lda_identity.npy', lda_identity)
np.save('unique_lasers.npy', unique_lasers)
np.save('taste_cosine_similarity.npy', taste_cosine_similarity)
np.save('taste_euclidean_distance.npy', taste_euclidean_distance)
#np.save('taste_mahalanobis_distance.npy', taste_mahalanobis_distance)
np.save('p_discriminability.npy', p_discriminability)

# Plot the r_squared values together first (for the different laser conditions)
fig = plt.figure()
for i in range(r_pearson.shape[0]):
	plt.plot(x[plot_indices], np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Pearson $r^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
plt.xlabel('Time from stimulus (ms)')
plt.ylabel('Average Pearson $r^2$')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Pearson correlation-palatability.png', bbox_inches = 'tight')
plt.close('all')

fig = plt.figure()
for i in range(r_spearman.shape[0]):
	plt.plot(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Spearman $rho^2$ with palatability ranks' + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
plt.xlabel('Time from stimulus (ms)')
plt.ylabel('Average Spearman $rho^2$')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Spearman correlation-palatability.png', bbox_inches = 'tight')
plt.close('all')

# Now plot the r_squared values separately for the different laser conditions
for i in range(r_pearson.shape[0]):
	fig = plt.figure()
	plt.errorbar(x[plot_indices], np.mean(r_pearson[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_pearson[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_pearson.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Pearson $r^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))	
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Pearson $r^2$')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Pearson correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all')

for i in range(r_spearman.shape[0]):
	fig = plt.figure()
	plt.errorbar(x[plot_indices], np.mean(r_spearman[i, plot_indices, :]**2, axis = 1), yerr = np.std(r_spearman[i, plot_indices, :]**2, axis = 1)/np.sqrt(r_spearman.shape[2]), linewidth = 3.0, elinewidth = 0.8, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Spearman $rho^2$ with palatability ranks, laser condition %i' % (i+1) + '\n' + 'Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')
	plt.ylabel('Average Spearman $rho^2$')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Spearman correlation-palatability,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all')

# Now plot the p values together using the significance criterion specified by the user
# Make a final p array - this will store 1s if x consecutive time bins have significant p values (these parameters are specified by the user)
p_pearson_final = np.zeros(p_pearson.shape)
p_spearman_final = np.zeros(p_spearman.shape)
p_identity_final = np.zeros(p_identity.shape)
for i in range(p_pearson_final.shape[0]):
	for j in range(p_pearson_final.shape[1]):
		for k in range(p_pearson_final.shape[2]):
			if (j < p_pearson_final.shape[1] - p_values[1]):
				if all(p_pearson[i, j:j + p_values[1], k] <= p_values[0]):
					p_pearson_final[i, j, k] = 1 
				if all(p_spearman[i, j:j + p_values[1], k] <= p_values[0]):
					p_spearman_final[i, j, k] = 1 
				if all(p_identity[i, j:j + p_values[1], k] <= p_values[0]):
					p_identity_final[i, j, k] = 1

# Also put the p_discriminability values together with the same rule as above
p_discriminability_final = np.zeros(p_discriminability.shape)
for i in range(p_discriminability.shape[0]):
	for j in range(p_discriminability.shape[1]):
		for k in range(p_discriminability.shape[2]):
			for l in range(p_discriminability.shape[3]):
				for m in range(p_discriminability.shape[4]):
					if (j < p_discriminability.shape[1] - p_values[1]):
						if all(p_discriminability[i, j:j + p_values[1], k, l, m] <= p_values[0]):
							p_discriminability_final[i, j, k, l, m] = 1.0

# Plot the p_discriminability values separately for each taste and laser condition
for i in range(p_discriminability_final.shape[0]):
	for j in range(p_discriminability_final.shape[2]):
		fig = plt.figure()
		for k in range(p_discriminability.shape[3]):
			plt.plot(x[plot_indices], np.mean(p_discriminability_final[i, plot_indices, j, k, :], axis = -1), linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.xlabel('Time from stimulus (ms)')	
		plt.ylabel('Fraction of significant neurons')
		plt.legend(loc = 'upper left', fontsize = 10)
		fig.savefig('Taste %i discriminability p values-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1]), bbox_inches = 'tight')
		plt.close("all")
		

# Now first plot the p values together for the different laser conditions
fig = plt.figure()
for i in range(p_pearson_final.shape[0]):
	plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
plt.xlabel('Time from stimulus (ms)')	
plt.ylabel('Fraction of significant neurons')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Pearson correlation p values-palatability.png', bbox_inches = 'tight')
plt.close('all') 

fig = plt.figure()
for i in range(p_spearman_final.shape[0]):
	plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
plt.xlabel('Time from stimulus (ms)')	
plt.ylabel('Fraction of significant neurons')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Spearman correlation p values-palatability.png', bbox_inches = 'tight')
plt.close('all')

fig = plt.figure()
for i in range(p_identity_final.shape[0]):
	plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
plt.xlabel('Time from stimulus (ms)')	
plt.ylabel('Fraction of significant neurons')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('ANOVA p values-identity.png', bbox_inches = 'tight')
plt.close('all')

# Now plot them separately for every laser condition
for i in range(p_pearson_final.shape[0]):
	fig = plt.figure()
	plt.plot(x[plot_indices], np.mean(p_pearson_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')	
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Pearson correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all') 

for i in range(p_spearman_final.shape[0]):
	fig = plt.figure()
	plt.plot(x[plot_indices], np.mean(p_spearman_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')	
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Spearman correlation p values-palatability,laser condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all')

for i in range(p_identity_final.shape[0]):
	fig = plt.figure()
	plt.plot(x[plot_indices], np.mean(p_identity_final[i, plot_indices, :], axis = 1), linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, threshold:%f, consecutive windows:%i' % (num_units, params[0][0], params[0][1], p_values[0], p_values[1]))
	plt.xlabel('Time from stimulus (ms)')	
	plt.ylabel('Fraction of significant neurons')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('ANOVA p values-identity,laser condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all')

# Now plot the LDA results for palatability and identity together for the different laser conditions
fig = plt.figure()
for i in range(lda_palatability.shape[0]):
	plt.plot(x[plot_indices], lda_palatability[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Units:%i, Window (ms):%i, Step (ms):%i, palatability LDA' % (num_units, params[0][0], params[0][1]))
plt.xlabel('Time from stimulus (ms)')	
plt.ylabel('Fraction correct trials (average across sessions)')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Palatability_LDA.png', bbox_inches = 'tight')
plt.close('all') 

fig = plt.figure()
for i in range(lda_identity.shape[0]):
	plt.plot(x[plot_indices], lda_identity[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
plt.title('Units:%i, Window (ms):%i, Step (ms):%i, identity LDA' % (num_units, params[0][0], params[0][1]))
plt.xlabel('Time from stimulus (ms)')	
plt.ylabel('Fraction correct trials (average across sessions)')
plt.legend(loc = 'upper left', fontsize = 10)
fig.savefig('Identity_LDA.png', bbox_inches = 'tight')
plt.close('all') 

# Now plot the LDA results separately for each laser condition
for i in range(lda_palatability.shape[0]):
	fig = plt.figure()
	plt.plot(x[plot_indices], lda_palatability[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, palatability LDA' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')	
	plt.ylabel('Fraction correct trials (average across sessions)')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Palatability_LDA,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all') 

for i in range(lda_identity.shape[0]):
	fig = plt.figure()
	plt.errorbar(x[plot_indices], lda_identity[i, plot_indices], linewidth = 3.0, label = 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
	plt.title('Units:%i, Window (ms):%i, Step (ms):%i, identity LDA' % (num_units, params[0][0], params[0][1]))
	plt.xlabel('Time from stimulus (ms)')	
	plt.ylabel('Fraction correct trials (average across sessions)')
	plt.legend(loc = 'upper left', fontsize = 10)
	fig.savefig('Identity_LDA,laser_condition%i.png' % (i+1), bbox_inches = 'tight')
	plt.close('all') 

# Plot the taste cosine similarity and distance plots for every laser condition and taste
# Start with cosine similarity
for i in range(taste_cosine_similarity.shape[0]):
	for j in range(taste_cosine_similarity.shape[2]):
		fig = plt.figure()
		for k in range(taste_cosine_similarity.shape[3]):
			plt.plot(x[plot_indices], taste_cosine_similarity[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.xlabel('Time from stimulus (ms)')	
		plt.ylabel('Average cosine similarity')
		plt.legend(loc = 'upper left', fontsize = 10)
		fig.savefig('Taste %i similarity values-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1]), bbox_inches = 'tight')
		plt.close("all")

# Now do the distances
for i in range(taste_euclidean_distance.shape[0]):
	for j in range(taste_euclidean_distance.shape[2]):
		fig = plt.figure()
		for k in range(taste_euclidean_distance.shape[3]):
			plt.plot(x[plot_indices], taste_euclidean_distance[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.xlabel('Time from stimulus (ms)')	
		plt.ylabel('Average Euclidean distance')
		plt.legend(loc = 'upper left', fontsize = 10)
		fig.savefig('Taste %i Euclidean distances-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1]), bbox_inches = 'tight')
		plt.close("all")

for i in range(pairwise_NB_identity.shape[0]):
	for j in range(pairwise_NB_identity.shape[2]):
		fig = plt.figure()
		for k in range(pairwise_NB_identity.shape[3]):
			plt.plot(x[plot_indices], pairwise_NB_identity[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.xlabel('Time from stimulus (ms)')	
		plt.ylabel('Average Pairwise Identity Accuracy (Naive Bayes)')
		plt.legend(loc = 'upper left', fontsize = 10)
		fig.savefig('Taste %i Pairwise Identity NB-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1]), bbox_inches = 'tight')
		plt.close("all")

'''
for i in range(taste_mahalanobis_distance.shape[0]):
	for j in range(taste_mahalanobis_distance.shape[2]):
		fig = plt.figure()
		for k in range(taste_mahalanobis_distance.shape[3]):
			plt.plot(x[plot_indices], taste_mahalanobis_distance[i, plot_indices, j, k], linewidth = 3.0, label = '%i vs %i' % (j+1, k+1))
		plt.title('Units:%i, Window (ms):%i, Step (ms):%i' % (num_units, params[0][0], params[0][1]) + '\n' + 'Dur:%ims, Lag:%ims' % (unique_lasers[0][i, 0], unique_lasers[0][i, 1]))
		plt.xlabel('Time from stimulus (ms)')	
		plt.ylabel('Average Mahalanobis distance')
		plt.legend(loc = 'upper left', fontsize = 10)
		fig.savefig('Taste %i Mahalanobis distances-Dur%i,Lag%i.png' % (j+1, unique_lasers[0][i, 0], unique_lasers[0][i, 1]), bbox_inches = 'tight')
		plt.close("all")
'''






	





























	
