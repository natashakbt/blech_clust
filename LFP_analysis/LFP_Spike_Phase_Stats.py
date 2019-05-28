#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:01:25 2019

@author: bradly
"""

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

# =============================================================================
# #Statistical running funtion
# def spike_phase_stats(data_frame,stats_frame,frequency_list,time_vector):
# 	for band_num in range(len(frequency_list)):
# 		for unit_num in	range(len(data_frame.unit.unique())):
# 			for time_num in range(len(time_vector)-1):
# 				
# 				#Set query to perfrom statistical tests on
# 				query = data_frame.query('band == @band_num and unit == @unit_num and time >= @time_vector[@time_num] and time < @time_vector[@time_num+1]')
# 				
# 				#Set query index
# 				queryidx = query.index
# 				
# 				#Ensures that idx is a possible boolean
# 				if len(queryidx) != 0:
# 	
# 					#Set query to perfrom statistical tests on
# 					stat_query = stats_frame.query('band == @band_num and unit == @unit_num and time_bin == @time_num')
# 					
# 					#Set stat query index
# 					statqueryidx = stat_query.index
# 					
# 					#Perform Rayleigh test of uniformity: H0 (null hypothesis): The population is distributed uniformly around the circle.
# 					stats_frame.Raytest_p[statqueryidx] = rayleightest(np.array(query.phase))
# 				
# 					#Make sure that this binned data is NOT distrubted uniformily around circle 
# 					#Discern where is the non-uniformity (radial location of Von Mises distribution) in radians (multiply by 57.2958, into degrees)
# 					#And the magnitude of this
# 					if rayleightest(np.array(query.phase)) <= 0.05:
# 						stats_frame.circ_mean[statqueryidx] = vonmisesmle(np.array(query.phase))[0]
# 						stats_frame.circ_kappa[statqueryidx]=vonmisesmle(np.array(query.phase))[1]
# 
# 
# 	return stats_frame
# 
# =============================================================================


def spike_phase_stats(data_frame,frequency_list,time_vector):
	#Create dataframe to store statistics with column for Rayleigh p values, mean circular angle of data, magnitude of non-uniformity, and distribution tests (comparitive)
	test_array = np.zeros((len(time_vector)-1,len(data_frame.taste.unique()),len(data_frame.unit.unique()),len(data_frame.band.unique())))
	
	nd_idx_objs = []
	for dim in range(test_array.ndim):
	    this_shape = np.ones(len(test_array.shape))
	    this_shape[dim] = test_array.shape[dim]
	    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(test_array.shape[dim]),this_shape.astype('int')), test_array.shape).flatten())
	
	stats_frame = pd.DataFrame(data = {'time_bin' : nd_idx_objs[0].flatten(),
	                                   'taste' : nd_idx_objs[1].flatten(),
	                                   'unit' : nd_idx_objs[2].flatten(),
	                                   'band' : nd_idx_objs[3].flatten()})
		
	stats_frame["Raytest_p"] = ""; stats_frame["circ_mean"] = ""; stats_frame["circ_kappa"] = ""

	
	for band_num in range(len(frequency_list)):
		for unit_num in	range(len(data_frame.unit.unique())):
			for time_num in range(len(time_vector)-1):
				
				#Set query to perfrom statistical tests on
				query = data_frame.query('band == @band_num and unit == @unit_num and time >= @time_vector[@time_num] and time < @time_vector[@time_num+1]')
				
				#Set query index
				queryidx = query.index
				
				#Ensures that idx is a possible boolean
				if len(queryidx) != 0:
	
					#Set query to perfrom statistical tests on
					stat_query = stats_frame.query('band == @band_num and unit == @unit_num and time_bin == @time_num')
					
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
	bins=300
else:
	#Change this dependending on the session type		
	t= np.linspace(0,1200000,50)
	bins=50
# =============================================================================
# 
# #Create dataframe to store statistics with column for Rayleigh p values, mean circular angle of data, magnitude of non-uniformity, and distribution tests (comparitive)
# test_array = np.zeros((len(t)-1,len(dframe.taste.unique()),len(dframe.unit.unique()),len(dframe.band.unique())))
# nd_idx_objs = []
# for dim in range(test_array.ndim):
#     this_shape = np.ones(len(test_array.shape))
#     this_shape[dim] = test_array.shape[dim]
#     nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(test_array.shape[dim]),this_shape.astype('int')), test_array.shape).flatten())
# 
# stats_dframe = pd.DataFrame(data = {'time_bin' : nd_idx_objs[0].flatten(),
#                                    'taste' : nd_idx_objs[1].flatten(),
#                                    'unit' : nd_idx_objs[2].flatten(),
#                                    'band' : nd_idx_objs[3].flatten()})
# 	
# stats_dframe["Raytest_p"] = ""; stats_dframe["circ_mean"] = ""; stats_dframe["circ_kappa"] = ""
# 
# =============================================================================
# =============================================================================
# #Parellel processing for statistics
# =============================================================================

#tempfunction to escape issue with using lambda
def tmpfunc(x, freq_bands=freq_bands, t=t):
	return spike_phase_stats(x,freq_bands,t)


#Perform statistics multiprocessing using taste as the group
dfnew = applyParallel(dframe.groupby(dframe.taste), tmpfunc)

#Clean up dataframe to account for multiple (* taste number) entries for values
df_ordered = dfnew
df_ordered.drop(df_ordered[df_ordered.taste > 0].index, inplace=True) #Drops all duplicates
df_ordered = df_ordered.reset_index(drop=True) #Resets the index to accurately detail info
df_ordered.taste = np.tile(dframe.taste.unique(),(len(t)-1)*(len(dframe.band.unique()))*len(dframe.unit.unique())*len(dframe.taste.unique())) #Reenters labels

#Save dframe into node within HdF5 file
df_ordered.to_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe')



#TRY PLOTTING
conv_cols = df_ordered.apply(pd.to_numeric, errors = 'coerce')

hist = conv_cols.hist(column='Raytest_p',bins=100)

query_check = conv_cols.query('band == 0 and unit == 0 and taste == 0')

hist = query_check.hist(column='Raytest_p',bins=25)

query_check = dframe.query('band == 0 and unit == 0 and taste == 0')
hist = query_check.hist(column='phase',bins=25)

colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))

for unit in dframe.unit.unique():
	
	#plot both conditions for unit
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
	fig.text(0.07, 0.5,'Number of Spikes', va='center', rotation='vertical',fontsize=14)
	fig.text(0.5, 0.05, 'Taste', ha='center',fontsize=14)
	axes_list = [item for sublist in axes for item in sublist]
	
	for ax, taste, color in zip(axes.flatten(),dframe.taste.unique(),colors):
		query_check = dframe.query('band == 0 and unit == @unit and taste == @taste and time>=2000 and time<=4000')
		df_var = query_check.phase
		
		ax = axes_list.pop(0)
		im =ax.hist(np.array(df_var), bins=25, color=color, alpha=0.7)
		ax.set_title(taste,size=15,y=1)
					
	
			
	fig.suptitle('Unit %i' %(dframe.unit.unique()[unit]),size=16)
							
	
	
	
	
				for ax,cond in zip(axes.flatten(),range(len(densities_joined))):
					
					im = ax.imshow(np.array(densities_joined[cond]).astype('double').T,cmap='inferno',aspect=.25,vmin=0, vmax=round(clim_set,1)) #THIS SETS THE COLORBAR LIMITS
					ax.set_title(dframe.condition.unique()[cond],size=15,y=1.08)
					ax.set_yticks(np.linspace(0,bins-1,5))
					ax.set_yticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
					ax.set_xticks(np.linspace(0,bins-2,6))
					ax.set_xticklabels(np.arange(0,24,4))













for taste in range(len(dframe.taste.unique())):
	for band in range (len(freq_bands)):
		for unit in 	range(len(dframe.unit.unique())):
			x1_all=[[None] *(bins) for i in range(len(t)-1)];x2_all=[[None] *(bins) for i in range(len(t)-1)];
			density1_all=[[None] *(bins) for i in range(len(t)-1)] ;density2_all=[[None] *(bins) for i in range(len(t)-1)]
			#chi_stat = [[None] *(bins) for i in range(len(t)-1)]; chi_p = [[None] *(bins) for i in range(len(t)-1)]
			mwp =[]; chi = []; kuiper_p = []
			for time in range(len(t)-1):
				idx1 = 	(dframe["taste"]==sorted(dframe.taste.unique())[taste]) & \
				(dframe["band"]==sorted(dframe.band.unique())[band]) & \
				(dframe["unit"]==int(dframe.unit.unique()[unit])) & \
				(dframe["time"]>=t[time]) & (dframe["time"]<t[time]+t[time+1])
				
# =============================================================================
# 				idx2 =	(dframe["band"]==sorted(dframe.band.unique())[band]) & \
# 				(dframe["unit"]==int(sorted_day2[animal][unit])) & \
# 				(dframe["time"]>=t[time]) & (dframe["time"]<t[time]+t[time+1])
# 				
# =============================================================================
				#Perform Rayleigh test of uniformity: H0 (null hypothesis): The population is distributed uniformly around the circle.
				stats_dframe.Raytest_p[idx1] = rayleightest(np.array(dframe.phase[idx1]))
				#stats_dframe.Raytest_p[idx2] = rayleightest(np.array(dframe.phase[idx2]))
								
				#Make sure that this binned data is NOT distrubted uniformily around circle 
				#Discern where is the non-uniformity (radial location of Von Mises distribution) in radians (multiply by 57.2958, into degrees)
				#And the magnitude of this
				if np.array(stats_dframe.Raytest_p[idx1])[0] <= 0.05:
					stats_dframe.circ_mean[idx1] = vonmisesmle(np.array(dframe.phase[idx1]))[0]
					stats_dframe.circ_kappa[idx1]=vonmisesmle(np.array(dframe.phase[idx1]))[1]
				#if np.array(stats_dframe.Raytest_p[idx2])[0] <= 0.05:
				#	stats_dframe.circ_mean[idx2] = vonmisesmle(np.array(dframe.phase[idx2]))[0]
				#	stats_dframe.circ_kappa[idx2]=vonmisesmle(np.array(dframe.phase[idx2]))[1]
					
# =============================================================================
# 				#Perform kuiper test on datasets
# 				p,k,K = circ_stats_stone.circular_kuipertest_stone(np.array(dframe.phase[idx1]),np.array(dframe.phase[idx2]),100,0)
# 				stats_dframe.kuiper_p[idx1] = p; stats_dframe.kuiper_p[idx2] = p
# 				stats_dframe.kuiper_kstat[idx1] = k; stats_dframe.kuiper_kstat[idx2] = k
# 				stats_dframe.kuiper_kcrit[idx1] = K; stats_dframe.kuiper_kcrit[idx2] = K
# 				kuiper_p.append(p)
# =============================================================================
		
for taste in range(len(dframe.taste.unique())):
	for trial in range(len(dframe.trial.unique())):
		for band in range (len(freq_bands)):
			for unit in 	range(len(dframe.unit.unique())):
				x1_all=[[None] *(bins) for i in range(len(t)-1)];x2_all=[[None] *(bins) for i in range(len(t)-1)];
				density1_all=[[None] *(bins) for i in range(len(t)-1)] ;density2_all=[[None] *(bins) for i in range(len(t)-1)]
				mwp =[]; chi = []; kuiper_p = []
				for time in range(len(t)-1):
					idx1 = 	(dframe["taste"]==sorted(dframe.taste.unique())[taste]) & \
					(dframe["trial"]==sorted(dframe.trial.unique())[trial]) & \
					(dframe["band"]==sorted(dframe.band.unique())[band]) & \
					(dframe["unit"]==int(dframe.unit.unique()[unit])) & \
					(dframe["time"]>=t[time]) & (dframe["time"]<t[time+1])
					
					#Ensures that idx is a possible boolean
					if idx1.sum() != 0:
		# =============================================================================
		# 				idx2 =	(dframe["band"]==sorted(dframe.band.unique())[band]) & \
		# 				(dframe["unit"]==int(sorted_day2[animal][unit])) & \
		# 				(dframe["time"]>=t[time]) & (dframe["time"]<t[time]+t[time+1])
		# 				
		# =============================================================================
						#Perform Rayleigh test of uniformity: H0 (null hypothesis): The population is distributed uniformly around the circle.
						stats_dframe.Raytest_p[idx1] = rayleightest(np.array(dframe.phase[idx1]))
						#stats_dframe.Raytest_p[idx2] = rayleightest(np.array(dframe.phase[idx2]))
										
						#Make sure that this binned data is NOT distrubted uniformily around circle 
						#Discern where is the non-uniformity (radial location of Von Mises distribution) in radians (multiply by 57.2958, into degrees)
						#And the magnitude of this
						if np.array(stats_dframe.Raytest_p[idx1])[0] <= 0.05:
							stats_dframe.circ_mean[idx1] = vonmisesmle(np.array(dframe.phase[idx1]))[0]
							stats_dframe.circ_kappa[idx1]=vonmisesmle(np.array(dframe.phase[idx1]))[1]



#dfnew = applyParallel(dframe.groupby(dframe.taste), spike_phase_stats(dframe,stats_dframe,freq_bands,t))




#NO TIME COMPONENT
for taste in range(len(dframe.taste.unique())):
	for trial in range(len(dframe.trial.unique())):
		for band in range (len(freq_bands)):
			for unit in 	range(len(dframe.unit.unique())):
				x1_all=[[None] *(bins) for i in range(len(t)-1)];x2_all=[[None] *(bins) for i in range(len(t)-1)];
				density1_all=[[None] *(bins) for i in range(len(t)-1)] ;density2_all=[[None] *(bins) for i in range(len(t)-1)]
				#chi_stat = [[None] *(bins) for i in range(len(t)-1)]; chi_p = [[None] *(bins) for i in range(len(t)-1)]
				mwp =[]; chi = []; kuiper_p = []
				#for time in range(len(t)-1):
				idx1 = 	(dframe["taste"]==sorted(dframe.taste.unique())[taste]) & \
				(dframe["trial"]==sorted(dframe.trial.unique())[trial]) & \
				(dframe["band"]==sorted(dframe.band.unique())[band]) & \
				(dframe["unit"]==int(dframe.unit.unique()[unit])) 
				
					
	# =============================================================================
	# 				idx2 =	(dframe["band"]==sorted(dframe.band.unique())[band]) & \
	# 				(dframe["unit"]==int(sorted_day2[animal][unit])) & \
	# 				(dframe["time"]>=t[time]) & (dframe["time"]<t[time]+t[time+1])
	# 				
	# =============================================================================
				#Perform Rayleigh test of uniformity: H0 (null hypothesis): The population is distributed uniformly around the circle.
				stats_dframe.Raytest_p[idx1] = rayleightest(np.array(dframe.phase[idx1]))
					#stats_dframe.Raytest_p[idx2] = rayleightest(np.array(dframe.phase[idx2]))
									
					#Make sure that this binned data is NOT distrubted uniformily around circle 
					#Discern where is the non-uniformity (radial location of Von Mises distribution) in radians (multiply by 57.2958, into degrees)
					#And the magnitude of this
				if np.array(stats_dframe.Raytest_p[idx1])[0] <= 0.05:
					stats_dframe.circ_mean[idx1] = vonmisesmle(np.array(dframe.phase[idx1]))[0]
					stats_dframe.circ_kappa[idx1]=vonmisesmle(np.array(dframe.phase[idx1]))[1]
					
					
					
					
					
					
				density1 = stats.gaussian_kde(dframe.phase[idx1]); 
				density2 = stats.gaussian_kde(dframe.phase[idx2])
				n1,x1=np.histogram(dframe.phase[idx1], bins=np.linspace(-np.pi,np.pi,bins), density=True);
				n2,x2=np.histogram(dframe.phase[idx2], bins=np.linspace(-np.pi,np.pi,bins), density=True);
				
				density1_all[time]=density1(x1); density2_all[time]=density2(x2)
				
			#combine density info
			densities_joined= [density1_all,density2_all]
			clim_set= np.max(densities_joined[:])
			
			#create sig_value vector
			sig_vector =np.tile(np.array(np.array(kuiper_p)<=0.05),(bins,1));
			if sum(np.array(np.array(kuiper_p)<=0.05))!=0:
				sig_tail = "yes"
		
				#plot both conditions for unit
				fig,axes = plt.subplots(nrows=2, ncols=1,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
				fig.text(0.07, 0.5,'Period', va='center', rotation='vertical',fontsize=14)
				fig.text(0.5, 0.12, 'Time (min)', ha='center',fontsize=14)
				axes_list = [item for sublist in axes for item in sublist]
				
				for ax,cond in zip(axes.flatten(),range(len(densities_joined))):
					ax = axes_list.pop(0)
					im = ax.imshow(np.array(densities_joined[cond]).astype('double').T,cmap='inferno',aspect=.25,vmin=0, vmax=round(clim_set,1)) #THIS SETS THE COLORBAR LIMITS
					ax.set_title(dframe.condition.unique()[cond],size=15,y=1.08)
					ax.set_yticks(np.linspace(0,bins-1,5))
					ax.set_yticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
					ax.set_xticks(np.linspace(0,bins-2,6))
					ax.set_xticklabels(np.arange(0,24,4))
					
					#Add the color bar
					if cond==0:
						cbar_ax = fig.add_axes([.92, 0.2, 0.01, 0.6])
						clb =fig.colorbar(im, cax=cbar_ax)	
					
					# Loop over data dimensions and create text annotations.
					for i in range(1):
					    for j in range(len(densities_joined[cond][0])-1):
							   if sig_vector[i, j]==True:
								   text1 = ax.text(j, i, '*',
						                       ha="center", va="center", color="black",size=18,bbox=dict(facecolor='red', alpha=0.5))
								   			
				clb.set_label('Phasic Density',size=14,labelpad=20,rotation=270)
				fig.subplots_adjust(hspace=0,wspace = 0.05)
				fig.suptitle('%s: Units: %i- %i' %(dframe.animal.unique()[animal],int(sorted_day1[animal][unit]),int(sorted_day2[animal][unit])) +'\n' + 'Freq. Band: %s' %(freq_bands[band]),size=16)
				
				fig.savefig(dir_name+'/%s_%s_Passive_units_%i_%i_%s.png'% (dframe.animal.unique()[animal],freq_bands[band],int(sorted_day1[animal][unit]),int(sorted_day2[animal][unit]),sig_tail))
				plt.close(fig)	
			
			else:
				sig_tail = "no"
		
				#plot both conditions for unit
				fig,axes = plt.subplots(nrows=2, ncols=1,sharex=True, sharey=True,figsize=(12, 8), squeeze=False)
				fig.text(0.07, 0.5,'Period', va='center', rotation='vertical',fontsize=14)
				fig.text(0.5, 0.12, 'Time (min)', ha='center',fontsize=14)
				axes_list = [item for sublist in axes for item in sublist]
				
				for ax,cond in zip(axes.flatten(),range(len(densities_joined))):
					ax = axes_list.pop(0)
					im = ax.imshow(np.array(densities_joined[cond]).astype('double').T,cmap='inferno',aspect=.25,vmin=0, vmax=round(clim_set,1)) #THIS SETS THE COLORBAR LIMITS
					ax.set_title(dframe.condition.unique()[cond],size=15,y=1.08)
					ax.set_yticks(np.linspace(0,bins-1,5))
					ax.set_yticklabels([r"-$\pi$",r"-$\pi/2$","$0$",r"$\pi/2$",r"$\pi$"])
					ax.set_xticks(np.linspace(0,bins-2,6))
					ax.set_xticklabels(np.arange(0,24,4))
					
					#Add the color bar
					if cond==0:
						cbar_ax = fig.add_axes([.92, 0.2, 0.01, 0.6])
						clb =fig.colorbar(im, cax=cbar_ax)	
					
					# Loop over data dimensions and create text annotations.
					for i in range(1):
					    for j in range(len(densities_joined[cond][0])-1):
							   if sig_vector[i, j]==True:
								   text1 = ax.text(j, i, '*',
						                       ha="center", va="center", color="black",size=18,bbox=dict(facecolor='red', alpha=0.5))
								   			
				clb.set_label('Phasic Density',size=14,labelpad=20,rotation=270)
				fig.subplots_adjust(hspace=0,wspace = 0.05)
				fig.suptitle('%s: Units: %i- %i' %(dframe.animal.unique()[animal],int(sorted_day1[animal][unit]),int(sorted_day2[animal][unit])) +'\n' + 'Freq. Band: %s' %(freq_bands[band]),size=16)
				
				fig.savefig(dir_name+'/%s_%s_Passive_units_%i_%i_%s.png'% (dframe.animal.unique()[animal],freq_bands[band],int(sorted_day1[animal][unit]),int(sorted_day2[animal][unit]),sig_tail))
				plt.close(fig)	
					
	print('Animal %s is done...' %(dframe.animal.unique()[animal]))
	
#put into dump file
tuple_save = 'Passive_Phaselock_stats_%i_%s_%i_%s.dump' %(len(all_day1), stats_dframe.condition.unique()[0],len(all_day2), stats_dframe.condition.unique()[1])
output_name =   os.path.join(dir_name, tuple_save)
pickle.dump(stats_dframe, open(output_name, 'wb'))  	

		