#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:48:04 2018

@author: bradly
"""
#Import necessary tools
import numpy as np
import easygui
import tables
import os
import scipy.io as sio
import matplotlib.pyplot as plt

#Get name of directory where marginal.py file is located
marg_dir_name = easygui.diropenbox(msg='Direct system to Marginal.py location', title='Marginal.py Locator')
os.chdir(marg_dir_name)

#Look for the marginal.py file in the directory
file_list = os.listdir('./')
marinal_name = ''
for files in file_list:
	if files[-2:] == 'py':
		marinal_name = files

import marginal

#Get name of directory where the data files and hdf5 file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

#Open the hdf5 file and create list of child paths
hf5 = tables.open_file(hdf5_name, 'r+')

#Read in LFP info
dig_in_channels = hf5.list_nodes('/digital_in')
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

# Ask if this is for drug or saline
msg   = "Are you using dig_in_4 (Saline) or dig_in_5 (Drug)?"
LFP_dig_in = easygui.buttonbox(msg,choices = ["4","5"])

#Extract data and squeeze (this is only one trial)
exec("LFP_array = hf5.root.Parsed_LFP.dig_in_%i_LFPs[:] " % int(LFP_dig_in[0]))
data = np.squeeze(LFP_array, axis=1)

#Set parameter specifications for windowing and frequencies
#Specify filtering parameters
Boxes = ['Sampling Frequency','Max','Min','Sampling Window','Sampling Overlap (default 90%)']
freqparam = easygui.multenterbox('Specify sampling parameters','Parameter Input', Boxes, values = ['1000' , '20', '3','500','450']) 
frequencies = np.arange(int(freqparam[2]), int(freqparam[1])+1, 1)

#Create time array using sampling rate (default: 1000Hz)
t = np.arange(data.shape[1])/int(freqparam[0])

#flip through channels and perform inference using binned windows
bin_overlap = int(freqparam[3])-int(freqparam[4])
full_inf_array = np.zeros((data.shape[0],int((data.shape[1]-int(freqparam[4]))/bin_overlap), len(frequencies)))
for channel in range(data.shape[0]):
    start_bin = 0
    for bin_win in range(int((data.shape[1]-int(freqparam[4]))/bin_overlap)):
        
        # Make a sinusoidal marginal model
        model = marginal.SinusoidMarginal()
        
        # Perform inference and store in array
        model.fit(data[channel,start_bin:start_bin+int(freqparam[3])], t[start_bin:start_bin+int(freqparam[3])], frequencies)
        full_inf_array[channel,bin_win,:] = model.posterior/np.sum(model.posterior)
        start_bin = start_bin+bin_overlap
        
#Using spec feature
#dt = 0.0005
#Fs = int(1.0/dt)  # the sampling frequency
#NFFT = 1024       # the length of the windowing segments
#
#plt.specgram(data, NFFT=NFFT, Fs=int(freqparam[0]), noverlap=900)
#
#for channel in range(data.shape[0]):
#     full_inf_array_spec[channel,:,:] =   
#    
#x = np.arange(int((data.shape[1]-int(freqparam[4]))/bin_overlap))

# Plot the posterior probabilities against frequencies
#plt.pcolor(a[0,:2000,:].T); plt.yticks(np.arange(18), frequencies); plt.colorbar()        
        
plt.plot(frequencies, model.posterior/np.sum(model.posterior))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Marginal posterior probability")
