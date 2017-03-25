import matplotlib
matplotlib.use('Agg')

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt

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

# Get the digital inputs/tastes
trains_dig_in = hf5.list_nodes('/spike_trains')

# Get the pre stimulus time from the hdf5 file
pre_stim = int(hf5.root.ancillary_analysis.pre_stim.read())

# Delete the directory for storing HMM-EMG plots if it exists, and make a new one
try:
	os.system('rm -r ./HMM_EMG_plots')
except:
	pass
os.mkdir('./HMM_EMG_plots')

# Pull out the EMG BSA results
gapes = hf5.root.ancillary_analysis.gapes[:]
ltps = hf5.root.ancillary_analysis.ltps[:]

# Pull out the gapes according to Li et al. 2016
gapes_Li = hf5.root.ancillary_analysis.gapes_Li[:]

# Pull out the significant trials on EMG
sig_trials = hf5.root.ancillary_analysis.sig_trials[:]

# Pull outthe laser conditions and the trials organized by laser condition
lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
trials = hf5.root.ancillary_analysis.trials[:]

# Run through the digital inputs
for dig_in in trains_dig_in:

	# Get the taste number
	taste_num = int(str.split(dig_in._v_pathname, '/')[-1][-1])
	
	# Make a directory for this digital input
	os.mkdir('./HMM_EMG_plots/dig_in_{:d}'.format(taste_num))

	# First check if this digital input has multinomial_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/multinomial_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/multinomial_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/multinomial_hmm_results/laser'.format(taste_num))
				
				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time 
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
						
						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/multinomial/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time 
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
					
					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Go back to the data directory
				os.chdir(dir_name)

	# Now check if this digital input has generic_poisson_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/generic_poisson_hmm_results/laser'.format(taste_num))
				
				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time 
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
						
						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/generic_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time 
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
					
					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Go back to the data directory
				os.chdir(dir_name)

	# Now check if this digital input has feedforward_poisson_hmm_results
	if hf5.__contains__('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results'.format(taste_num)):
		# If it does, then make a folder for multinomial hmm plots
		os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson'.format(taste_num))

		# List the nodes under multinomial_hmm_results
		hmm_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results'.format(taste_num))

		# Run through the hmm_nodes, make folders for each of them, and plot the posterior probabilities
		for node in hmm_nodes:
			# Check if the current node is the laser node
			if str.split(node._v_pathname, '/')[-1] == 'laser':
				# Get the nodes with the laser results
				laser_nodes = hf5.list_nodes('/spike_trains/dig_in_{:d}/feedforward_poisson_hmm_results/laser'.format(taste_num))
				
				# Run through the laser_nodes, make folders for each of them, and plot the posterior probabilities
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser'.format(taste_num))
				for laser_node in laser_nodes:
					# Make a folder for this node
					os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))
					# Change to this directory
					os.chdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/laser/{:s}'.format(taste_num, str.split(laser_node._v_pathname, '/')[-1]))

					# Get the HMM time 
					time = laser_node.time[:]
					# And the posterior probability to plot
					posterior_proba = laser_node.posterior_proba[:]

					# Get the limits of plotting
					start = 100*(int(time[0]/100))
					end = 100*(int(time[-1]/100) + 1)

					# Make directories for the plots
					os.mkdir('./gapes')
					os.mkdir('./ltps')
					# Make folders by laser conditions too
					for condition in lasers:
						os.mkdir('./gapes/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
						os.mkdir('./ltps/Dur%i,Lag%i' % (int(condition[0]), int(condition[1])))
					# Run through the trials
					for i in range(posterior_proba.shape[0]):
						# Locate this trial number in the lasers X trial X.. array called trials
						laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
						this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
						this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
						
						# Plot the gapes, gapes_Li and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
							plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./gapes/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

						# Plot the ltps, and posterior_proba
						fig = plt.figure()
						if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
							plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
						for j in range(posterior_proba.shape[2]):
							plt.plot(time, posterior_proba[i, :, j])
						plt.xlabel('Time post stimulus (ms)')
						plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
						plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations[i], dig_in.laser_onset_lag[i]))
						fig.savefig('./ltps/Dur%i,Lag%i/Trial_%i.png' % (int(lasers[laser_condition, 0]), int(lasers[laser_condition, 1]), i+1))
						plt.close("all")

					# Go back to the data directory
					os.chdir(dir_name)

			else:
				# Make a folder for this node
				os.mkdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Change to this directory
				os.chdir('./HMM_EMG_plots/dig_in_{:d}/feedforward_poisson/{:s}'.format(taste_num, str.split(node._v_pathname, '/')[-1]))
				# Get the HMM time 
				time = node.time[:]
				# And the posterior probability to plot
				posterior_proba = node.posterior_proba[:]

				# Get the limits of plotting
				start = 100*(int(time[0]/100))
				end = 100*(int(time[-1]/100) + 1)

				# Make directories for the plots
				os.mkdir('./gapes')
				os.mkdir('./ltps')
				# Run through the trials
				for i in range(posterior_proba.shape[0]):
					# Locate this trial number in the lasers X trial X.. array called trials
					laser_condition = int(np.where(trials == posterior_proba.shape[0]*taste_num + i)[0][0])
					this_taste_trials = np.where((trials[laser_condition] >= posterior_proba.shape[0]*taste_num) * (trials[laser_condition] <= posterior_proba.shape[0]*(taste_num + 1)))
					this_trial = int(np.where(trials[laser_condition, this_taste_trials][0] == posterior_proba.shape[0]*taste_num + i)[0])
					
					# Plot the gapes, gapes_Li and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), gapes[laser_condition, taste_num, this_trial, :end])
						plt.plot(np.arange(end), gapes_Li[laser_condition, taste_num, this_trial, pre_stim : pre_stim + end], linewidth = 2.0, color = 'black')
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power < 4.6Hz, Gapes from Li et al')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./gapes/Trial_%i.png' % (i+1))
					plt.close("all")

					# Plot the ltps, and posterior_proba
					fig = plt.figure()
					if sig_trials[laser_condition, taste_num, this_trial] > 0.0:
						plt.plot(np.arange(end), ltps[laser_condition, taste_num, this_trial, :end])
					for j in range(posterior_proba.shape[2]):
						plt.plot(time, posterior_proba[i, :, j])
					plt.xlabel('Time post stimulus (ms)')
					plt.ylabel('Probability of HMM states' + '\n' + '% Power in 5.95-8.6Hz')
					plt.title('Trial %i' % (i+1))
					fig.savefig('./ltps/Trial_%i.png' % (i+1))
					plt.close("all")

				# Go back to the data directory
				os.chdir(dir_name)





hf5.close()
				
				
	

	

	






