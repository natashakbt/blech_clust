# Create plots for filtered emg and respective envelope

# Import stuff
import numpy as np
from scipy.signal import butter, filtfilt, periodogram
import easygui
import os
import pylab as plt

# Ask for the directory where the data (emg_data.npy) sits
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#emg_data = np.load('emg_data.npy')
emg_filt = np.load('emg_filt.npy')
emg_env = np.load('env.npy')
sig_trials = np.load('sig_trials.npy')

# Ask the user for the directory to save plots etc in
dir_name = easygui.diropenbox(msg = 'Choose the output directory for EMG BSA analysis')
os.chdir(dir_name)

# Ask the user for the names of the tastes in the dataset
tastes = easygui.multenterbox(
        msg = 'Enter the names of the tastes used in the experiments', 
        fields = ['Taste{:d}'.format(i+1) for i in range(emg_env.shape[0])])

time_limits = easygui.multenterbox(
        msg = 'Time limits for plotting [relative to stim delivery]', 
        fields = ['Stim Delivery (ms)', 'Pre stim (ms)', 'Post stim (ms)'])
time_limits = [int(x) for x in time_limits]
fin_time_limits = time_limits[1:]
fin_inds = [x + time_limits[0] for x in time_limits[1:]]
time_vec = np.arange(*fin_time_limits)

colors = ['r','b']

cut_emg_filt = emg_filt[...,fin_inds[0]:fin_inds[1]]
cut_emg_env = emg_env[...,fin_inds[0]:fin_inds[1]]

fig,ax = plt.subplots(*emg_filt.shape[:2][::-1], 
        sharey=True, sharex=True, figsize = (20,30))
inds = list(np.ndindex(ax.shape))
for trial, taste in inds:
    this_color = colors[int(sig_trials[taste,trial])]
    ax[trial,taste].plot(time_vec, cut_emg_filt[taste,trial], 
            c = this_color)
    if trial == 0:
        this_taste = tastes[taste]
        ax[trial,taste].set_title(this_taste)
    if trial == emg_filt.shape[1]-1:
        ax[trial,taste].set_xlabel('Time post-sitm (ms)')
fig.savefig('emg_filtered_plots.png', bbox_inches = 'tight')
plt.close(fig)
#plt.show()

fig,ax = plt.subplots(*emg_filt.shape[:2][::-1], 
        sharey=True, sharex=True, figsize = (20,30))
inds = list(np.ndindex(ax.shape))
for trial, taste in inds:
    this_color = colors[int(sig_trials[taste,trial])]
    ax[trial,taste].plot(time_vec, cut_emg_env[taste,trial], 
            c = this_color)
    if trial == 0:
        this_taste = tastes[taste]
        ax[trial,taste].set_title(this_taste)
    if trial == emg_filt.shape[1]-1:
        ax[trial,taste].set_xlabel('Time post-sitm (ms)')
fig.savefig('emg_envelope_plots.png', bbox_inches = 'tight')
plt.close(fig)
plt.show()
