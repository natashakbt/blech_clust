#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:01:25 2019

@author: bradly
"""

# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    

#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

#import tools for multiprocessing
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed, parallel_backend #for parallel processing

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import matplotlib.pyplot as plt # makes matplotlib work like MATLAB. ’pyplot’ functions.
from scipy.stats import sem
from scipy import stats
from scipy.stats import chi2_contingency
from pycircstat import descriptive, swap2zeroaxis 
import math
import sys
import easygui
import tables
from matplotlib.ticker import PercentFormatter
from pylab import text
import pylab as P
from astropy.stats import rayleightest
from astropy.stats import circmean
from astropy.stats import vonmisesmle
from astropy import units as u
from pycircstat import kuiper


# =============================================================================
# #append directory to load circular statistic codes
# =============================================================================

sys.path.append('/home/bradly/anaconda3/lib/python3.6/stone_circ_stats')
from stone_circ_stats import circ_stats_stone

# =============================================================================
# #Define functions used in code
# =============================================================================

#Parellel processing function
def applyParallel(dfGrouped, func, parallel_kws={}, backend='multiprocessing', backend_kws={}):
    ''' Parallel version of pandas apply '''

    if not isinstance(dfGrouped, pd.core.groupby.GroupBy):
        raise TypeError(f'dfGrouped must be pandas.core.groupby.GroupBy, not {type(dfGrouped)}')

    # Set default parallel args
    default_parallel_kws = dict(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=11)
    for key,item in default_parallel_kws.items():
        parallel_kws.setdefault(key, item)
    print("Apply parallel with {} verbosity".format(parallel_kws["verbose"]))

    # Compute
    with parallel_backend(backend, **backend_kws): # backend decides how job lib will run your jobs, e.g. threads/processes/dask/etc
        retLst = Parallel(**parallel_kws)(delayed(func)(group) for name, group in dfGrouped)

    return pd.concat(retLst) # return concatonated result

 #Statistical running funtion
def spike_phase_stats(data_frame,frequency_list,time_vector):
        """
        Create dataframe to store statistics with column for Rayleigh p values, mean circular 
        angle of data, magnitude of non-uniformity, and distribution tests (comparitive)
        """

	frame_dict = {'time_bin' : np.asarray(time_vector[:-1]).astype(int),
			 'taste' : data_frame.taste.unique(),
			 'unit' : data_frame.unit.unique(),
			 'band' : data_frame.band.unique()}

	frame_list = [pd.DataFrame(data = val,columns=[key]) for key,val in frame_dict.items()]
	for frame in frame_list:
		frame['tmp'] = 1
		
	stats_frame = frame_list[0]
	for frame_num in range(1,len(frame_list)):
		stats_frame = stats_frame.merge(frame_list[frame_num],how='outer')
	stats_frame = stats_frame.drop(['tmp'],axis=1)
	
	stats_frame["Raytest_p"] = ""; stats_frame["circ_mean"] = ""; stats_frame["circ_kappa"] = ""



    ########################
    data_frame = dframe.copy()
	data_frame['time_bin'] = pd.cut(x=data_frame.time,bins=t,labels=np.asarray(t[:-1]).astype(int))
    test = data_frame.groupby(['band','taste','unit','time_bin'])
    
    x = test.get_group((0,0,0,0))
    

        
    ind_stats_frame = stats_frame.copy().set_index(['band','taste','unit','time_bin'])
    
    for name,group in test:
        ind_stats_frame.loc[name,['Raytest_p','circ_mean','circ_kappa']] = \
                                                    [rayleightest(np.array(group.phase)),
                                                     vonmisesmle(np.array(group.phase))[0],
                                                     vonmisesmle(np.array(group.phase))[1]
                                                    ]
    
    #########################
	for band_num in range(len(frequency_list)):
		for taste_num in	range(len(data_frame.taste.unique())):
			for time_num in range(len(time_vector)-1):
				
				#Set query to perfrom statistical tests on
				query = data_frame.query('band == @band_num  \
							 and taste == @taste_num \
							 and time >= @time_vector[@time_num] \
							 and time < @time_vector[@time_num+1]')
				
				#Set query index
				queryidx = query.index
				
				#Ensures that idx is a possible boolean
				if len(queryidx) != 0:
	
					#Set query to perfrom statistical tests on
					stat_query = stats_frame.query('band == @band_num \
									and taste == @taste_num \
									and time_bin == @time_vector[@time_num]')
					
					#Set stat query index
					statqueryidx = stat_query.index
					
					#Perform Rayleigh test of uniformity: H0 (null hypothesis): The population is distributed uniformly around the circle.
					stats_frame.Raytest_p[statqueryidx] = rayleightest(np.array(query.phase))
				
					#Make sure that this binned data is NOT distrubted uniformily around circle 
					#Discern where is the non-uniformity (radial location of Von Mises distribution) in radians (multiply by 57.2958, into degrees)
					#And the magnitude of this
					if rayleightest(np.array(query.phase)) <= 0.05:
						stats_frame.circ_mean[statqueryidx] = vonmisesmle(np.array(query.phase))[0]
						stats_frame.circ_kappa[statqueryidx]=vonmisesmle(np.array(query.phase))[1]


	return stats_frame

# ___                            _     ____        _        
#|_ _|_ __ ___  _ __   ___  _ __| |_  |  _ \  __ _| |_ __ _ 
# | || '_ ` _ \| '_ \ / _ \| '__| __| | | | |/ _` | __/ _` |
# | || | | | | | |_) | (_) | |  | |_  | |_| | (_| | || (_| |
#|___|_| |_| |_| .__/ \___/|_|   \__| |____/ \__,_|\__\__,_|
#              |_|                                          

# =============================================================================
# Import/Open HDF5 File and variables
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

#pull in dframes
dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/dframe','r+')
freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')

#Exctract frequency names
freq_bands = np.array(freq_dframe.iloc[:][0]).astype(str).reshape(np.array(freq_dframe.iloc[:][0]).size,1)

#Create time vector (CHANGE THIS BASED ON BIN SIZING NEEDS)		
if np.size(dframe.taste.unique())>0:
	#Change this dependending on the session type		
	t= np.linspace(0,7000,71)
	bins=70
else:
	#Change this dependending on the session type		
	t= np.linspace(0,1200000,50)
	bins=50

# ____                              _             
#|  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
#| |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#|  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#|_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                           |___/ 

# =============================================================================
# #Parellel processing for statistics
# =============================================================================

#tempfunction to escape issue with using lambda
def tmpfunc(x, freq_bands=freq_bands, t=t):
	return spike_phase_stats(x,freq_bands,t)

#Perform statistics multiprocessing using taste as the group
dfnew = applyParallel(dframe.groupby(dframe.unit), tmpfunc)

# =============================================================================
# #Store back into HDF5
# =============================================================================

#convert strings to numerical data
dfnew = dfnew.apply(pd.to_numeric, errors = 'coerce')

#Save dframe into node within HdF5 file
dfnew.to_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe')