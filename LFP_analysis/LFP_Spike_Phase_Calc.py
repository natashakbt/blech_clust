#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:23:58 2019

@author: abuzarmahmood/bradly

Created on Wed Feb 13 19:36:13 2019

"""
# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
from scipy.signal import hilbert #Hilbert transform to determine the amplitude envelope and instantaneous frequency of an amplitude-modulated signal
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
from tqdm import trange
import pandas as pd

# =============================================================================
# Define functions
# =============================================================================

#define bandpass filter parameters to parse out frequencies
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

# =============================================================================
# Define common variables
# =============================================================================
#specify frequency bands
iter_freqs = [
        ('Theta', 4, 7),
        ('Alpha', 7, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

#Covert to dframe for storing
freq_dframe = pd.DataFrame.from_dict(iter_freqs)

colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(iter_freqs)))

# =============================================================================
# Import/Open HDF5 File
# =============================================================================

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Pull LFPS and spikes
lfps_dig_in = hf5.list_nodes('/Parsed_LFP')
trains_dig_in = hf5.list_nodes('/spike_trains')
lfp_array = np.asarray([lfp[:] for lfp in lfps_dig_in])
spike_array = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])

# =============================================================================
# Processing
# =============================================================================
# =============================================================================
# Calculate phases
# =============================================================================

# Create processed phase arrays
analytic_signal_array = np.zeros((len(iter_freqs),) + lfp_array.shape, dtype = np.complex128)
phase_array = np.zeros((len(iter_freqs),) + lfp_array.shape)

for band in trange(len(iter_freqs), desc = 'bands'):
    for taste in range(lfp_array.shape[0]):
        for channel in range(lfp_array.shape[1]):
            for trial in range(lfp_array.shape[2]):
                band_filt_sig = butter_bandpass_filter(data = lfp_array[taste,channel,trial,:], 
                                                       lowcut = iter_freqs[band][1], 
                                                       highcut =  iter_freqs[band][2], 
                                                       fs = 1000)
                analytic_signal = hilbert(band_filt_sig)
                instantaneous_phase = np.angle(analytic_signal)
                
                analytic_signal_array[band,taste,channel,trial,:] = analytic_signal
                phase_array[band,taste,channel,trial,:] = instantaneous_phase
				   	
# =============================================================================
# Use mean LFP (across channels) to calculate phase (since all channels have same phase)
# =============================================================================
mean_analytic_signal = np.mean(analytic_signal_array,axis=2)
mean_phase_array = np.angle(mean_analytic_signal)

# =============================================================================
# Calculate phase locking: for every spike, find phase for every band
# =============================================================================
# Find spiketimes
# Find what phase each spike occured	
spike_times = spike_array.nonzero()
spikes_frame = pd.DataFrame(data = {'taste':spike_times[0],
                                    'trial':spike_times[1],
                                    'unit':spike_times[2],
                                    'time':spike_times[3]})

# Create array index identifiers
nd_idx_objs = []
for dim in range(mean_phase_array.ndim):
    this_shape = np.ones(len(mean_phase_array.shape))
    this_shape[dim] = mean_phase_array.shape[dim]
    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(mean_phase_array.shape[dim]),this_shape.astype('int')), mean_phase_array.shape).flatten())

phase_frame = pd.DataFrame(data = {'band' : nd_idx_objs[0].flatten(),
                                    'taste' : nd_idx_objs[1].flatten(),
                                    'trial' : nd_idx_objs[2].flatten(),
                                    'time' : nd_idx_objs[3].flatten(),
                                    'phase' : mean_phase_array.flatten()})

# Merge : Gives dataframe with length of (bands x numner of spikes)
final_phase_frame = pd.merge(spikes_frame,phase_frame,how='inner')

#Save dframes into node within HdF5 file
final_phase_frame.to_hdf(hdf5_name,'Spike_Phase_Dframe/dframe')
freq_dframe.to_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys')

#Flush and close file
hf5.flush()
hf5.close()
