#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:23:58 2019

@author: abuzarmahmood/bradly

Created on Wed Feb 13 19:36:13 2019

"""

# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    

# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
#Hilbert transform to determine the amplitude envelope and 
#instantaneous frequency of an amplitude-modulated signal
from scipy.signal import hilbert 
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
import easygui
import tables
from tqdm import trange
import pandas as pd
import collections

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

#Get name of directory where the data files and hdf5 file sits, 
#and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Load flagged channels from HDF5
# If absent, make empty data frame
try:
    flagged_channels = pd.read_hdf(hdf5_name,'/Parsed_LFP/flagged_channels')
except:
    print('No flagged channels dataset present. Defaulting to not using flags.')
    flagged_channels = pd.DataFrame()

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Pull LFPS and spikes
lfps_dig_in = hf5.list_nodes('/Parsed_LFP')
# Make sure not taking anything other than a dig_in
lfps_dig_in = [node for node in lfps_dig_in \
        if 'dig_in' in str(node)]
trains_dig_in = hf5.list_nodes('/spike_trains')

# Load LFPs and remove flagged channels if present
lfp_list = [lfp[:] for lfp in lfps_dig_in]

if len(flagged_channels) > 0:
    good_channel_list = [[channel for channel in range(len(lfp_list[dig_in])) \
        if channel not in \
        list(flagged_channels.query('Dig_In == {}'.format(dig_in))['Channel'])] \
        for dig_in in range(len(lfps_dig_in))]
else:
    good_channel_list = [[channel for channel in range(len(lfp_list[dig_in]))] \
        for dig_in in range(len(lfps_dig_in))]

lfp_list = [dig_in[good_channel_list[dig_in_num],:] \
        for dig_in_num,dig_in in enumerate(lfp_list)]
spike_array = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])

# ____                              _             
#|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
#| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                           |___/ 

# =============================================================================
# Calculate phases
# =============================================================================

# Create lists of phase array
# Save data as namedTuple so band and taste are annotated
filtered_tuple = collections.namedtuple('BandpassFilteredData',['Band','Taste','Data'])
filtered_signal_list = [ filtered_tuple (band, taste,
                                butter_bandpass_filter(
                                            data = dig_in,
                                            lowcut = iter_freqs[band][1], 
                                            highcut =  iter_freqs[band][2], 
                                            fs = 1000)) \
                                            for taste,dig_in in enumerate(lfp_list)\
                    for band in trange(len(iter_freqs), desc = 'bands') \
                    if len(dig_in) > 0]

# =============================================================================
# Use mean LFP (across channels) to calculate phase (since all channels have same phase)
# =============================================================================
# Process filtered signals to extract hilbert transform and phase 
mean_analytic_signal_list = [ filtered_tuple(x.Band, x.Taste, np.mean(hilbert(x.Data),axis=0)) \
                        for x in filtered_signal_list ]
mean_phase_list = [ filtered_tuple(x.Band, x.Taste, np.mean(np.angle(x.Data),axis=0)) \
                        for x in filtered_signal_list ]
                           
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
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

nd_idx_objs = make_array_identifiers(mean_phase_array)

phase_frame = pd.concat(
        [pd.DataFrame( data = { 'band' : dat.Band,
                                'taste' : dat.Taste,
                                'trial' : idx[0].flatten(),
                                'time' : idx[1].flatten(),
                                'phase' : dat.Data.flatten()}) \
                                for dat, idx in \
                    map(lambda dat: (dat, make_array_identifiers(dat.Data)),mean_phase_list)]
        )

# Merge : Gives dataframe with length of (bands x numner of spikes)
final_phase_frame = pd.merge(spikes_frame,phase_frame,how='inner')

#  ___        _               _   
# / _ \ _   _| |_ _ __  _   _| |_ 
#| | | | | | | __| '_ \| | | | __|
#| |_| | |_| | |_| |_) | |_| | |_ 
# \___/ \__,_|\__| .__/ \__,_|\__|
#                |_|              

#Save dframes into node within HdF5 file
final_phase_frame.to_hdf(hdf5_name,'Spike_Phase_Dframe/dframe')
freq_dframe.to_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys')

#Flush and close file
hf5.flush()
hf5.close()
