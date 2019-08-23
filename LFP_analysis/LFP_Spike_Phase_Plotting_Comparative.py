#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:07:34 2019

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
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
import re
import math
import easygui
import tables
from pylab import text
import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import gaussian_filter1d

# =============================================================================
# #Establish files to work on
# =============================================================================
#Get name of directory where you want to save output files to
save_name = easygui.diropenbox(msg = 'Choose directory you want output files' +\
							   'sent to (and/or where ".dir" files are)')

#Ask user if they have ran set-up before
msg   = "Have you performed directory set-up before (ie. do you have '.dir' files' in output folder) ?"
dir_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if dir_check == "No":
    # Ask the user for the hdf5 files that need to be plotted together (fist condition)
    dirs_1 = []
    while True:
    	dir_name = easygui.diropenbox(msg = 'Choose first condition directory with a hdf5 file, hit cancel to stop choosing')
    	try:
    		if len(dir_name) > 0:	
    			dirs_1.append(dir_name)
    	except:
    		break   

    #Dump the directory names into chosen output location for each condition
    #condition 1
    completeName_1 = os.path.join(save_name, 'dirs_cond1.dir') 
    f_1 = open(completeName_1, 'w')
    for item in dirs_1:
        f_1.write("%s\n" % item)
    f_1.close()


if dir_check == "Yes":
    #establish directories to flip through
    #condition 1
    dirs_1_path = os.path.join(save_name, 'dirs_cond1.dir')
    dirs_1_file = open(dirs_1_path,'r')
    dirs_1 = dirs_1_file.read().splitlines()
    dirs_1_file.close()
	
	#condition 1
    dirs_2_path = os.path.join(save_name, 'dirs_cond2.dir')
    dirs_2_file = open(dirs_2_path,'r')
    dirs_2 = dirs_2_file.read().splitlines()
    dirs_2_file.close()

# Make directory to store all phaselocking plots. Delete and 
# remake the directory if it exists
os.chdir(save_name)

try:
    os.system('rm -r '+'./Grouped_Phase_lock_analyses')
except:
    pass
os.mkdir('./Grouped_Phase_lock_analyses')

#Grab details regarding conditions from user
conditions = easygui.multenterbox(msg = 'Input condition names:', fields = ['Condition 1:','Condition 2:'], values = ['LiCl','Saline'])
	
#Create blank Dataframe for storing all animals
grouped_df=pd.DataFrame()

#Flip through files and extract dataframes and store into large one
for dir_name in dirs_1:	
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r+')	

	freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')
	dframe_stat = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe','r+')
		
	#Add column name for Animal and Condition and fill in appropriately
	dframe_stat.insert(0,'Animal',hdf5_name[0:4])
	dframe_stat.insert(1,'Condition',conditions[0])
	
	#Stack dataframes onto eachother
	grouped_df = pd.concat([grouped_df,dframe_stat],sort=False)
	
	#Close the hdf5 file
	hf5.close()

for dir_name in dirs_2:	
	#Change to the directory
	os.chdir(dir_name)
	#Locate the hdf5 file
	file_list = os.listdir('./')
	hdf5_name = ''
	for files in file_list:
		if files[-2:] == 'h5':
			hdf5_name = files

	#Open the hdf5 file
	hf5 = tables.open_file(hdf5_name, 'r+')	

	freq_dframe = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/freq_keys','r+')
	dframe_stat = pd.read_hdf(hdf5_name,'Spike_Phase_Dframe/stats_dframe','r+')
		
	#Add column name for Animal and Condition and fill in appropriately
	dframe_stat.insert(0,'Animal',hdf5_name[0:4])
	dframe_stat.insert(1,'Condition',conditions[1])
	
	#Stack dataframes onto eachother
	grouped_df = pd.concat([grouped_df,dframe_stat],sort=False)
	
	#Close the hdf5 file
	hf5.close()
	
#Ask user to clean up dataframe if need be
msg   = "Would you like to clean the dataframe up further (e.g. remove low firing units)?"
clean_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if clean_check == "Yes":
	clean_animal = easygui.multchoicebox(msg = 'Which Animals would you like to update (select as many as you want, or Cancel)?',
	choices = ([i for i in sorted(grouped_df.Animal.unique())]))
	
	if len(clean_animal)<0:
		print("I do not think you are understanding how this code works.")
		pass
	else:
		for animal in range(len(clean_animal)):
			clean_cond = easygui.multchoicebox(msg = 'Which condition would you like to update for %s?' %(clean_animal[animal]),
			choices = ([i for i in np.unique(sorted(grouped_df.Condition[grouped_df.Animal == clean_animal[animal]]))]))
			if len(clean_animal)<0:
				print("I do not think you are understanding how this code works.")
				pass
			else:
				#These are the real values to change/update
				for cond in  range(len(clean_cond)):
					remove_units = easygui.multchoicebox(msg = 'Which units would you like to remove for %s under the %s condition?' %(clean_animal[animal],clean_cond[cond]),
					choices = ([i for i in np.unique(sorted(grouped_df.unit[(grouped_df.Animal == clean_animal[animal]) & (grouped_df.Condition == clean_cond[cond])]))]))
					if len(remove_units)<0:
						print("I do not think you are understanding how this code works.")
						pass
					else:
						for unit_clear in range(len(remove_units)):
							grouped_df.drop(grouped_df[(grouped_df.Animal == clean_animal[animal]) & (grouped_df.Condition == clean_cond[cond]) & (grouped_df.unit == int(remove_units[unit_clear]))].index, inplace=True)
else:
	pass	


#Add categorical column to dataset
grouped_df['pval_cat'] = ''
grouped_df.loc[(grouped_df['Raytest_p'] > 0) & (grouped_df['Raytest_p'] <= 0.05),
        'pval_cat'] = int(0)
grouped_df.loc[(grouped_df['Raytest_p'] > 0.05) & (grouped_df['Raytest_p'] <= 0.1),
        'pval_cat'] = int(1)
grouped_df.loc[(grouped_df['Raytest_p'] > 0.1),'pval_cat'] = int(2)
grouped_df = grouped_df.replace('',np.nan)

# =============================================================================
# #Establish variables for processing
# =============================================================================
#Create time vector (CHANGE THIS BASED ON BIN SIZING NEEDS)     
if np.size(grouped_df.taste.unique())>0:
    # Set paramaters from the user
    params = easygui.multenterbox(msg = 'Enter the parameters for plotting', 
            fields = ['Pre stimulus spike train (ms)',
                    'Post-stimulus spike train (ms)', 
                    'Bin size (ms)','Pre stimulus bin (ms)',
                    'Post-stimulus bin (ms)'],
            values = ['2000','5000','50','2000','2000'])
    for i in range(len(params)):
        params[i] = int(params[i])  
    
    t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
    bins=params[2]
    
    identities = easygui.multenterbox(\
            msg = 'Put in the taste identities of the digital inputs', 
            fields = [tastant for tastant in range(len(grouped_df.taste.unique()))], 
            values=['NaCl','Sucrose','Citric Acid','QHCl'])
else:
    # Set paramaters from the user
    params = easygui.multenterbox(msg = 'Enter the parameters for plotting', 
            fields = ['Pre stimulus (ms)','Post-stimulus (ms)', 'Bin size (ms)'],
            values = ['0','1200000','50'])
    for i in range(len(params)):
        params[i] = int(params[i])  
    #Change this dependending on the session type       
    t= np.linspace(0,params[0]+params[1],((params[0]+params[1])//100)+1)
    bins=params[2]
    
#Exctract frequency names
freq_bands = np.array(freq_dframe.iloc[:][0]).astype(str).\
        reshape(np.array(freq_dframe.iloc[:][0]).size,1)
freq_vals = freq_dframe.to_dict('index')


# Make directory to store histogram plots. 
# Delete and remake the directory if it exists
#Change to the directory
os.chdir(save_name)
try:
	os.system('rm -r '+'./Grouped_Phase_lock_analyses/SPL_Distributions')
except:
        pass
os.mkdir('./Grouped_Phase_lock_analyses/SPL_Distributions')	

for band in sorted(grouped_df.band.unique()):
	colors_new = plt.get_cmap('RdBu')(np.linspace(0, 1, 4))
	fig,axes = plt.subplots(nrows=2, ncols=2,sharex=True, sharey=True,
	        figsize=(12, 8), squeeze=False)
	fig.text(0.075, 0.5,'Percentage of Units', va='center', 
	        rotation='vertical',fontsize=15)
	fig.text(0.5, 0.05, 'Time', ha='center',fontsize=15)
	axes_list = [item for sublist in axes for item in sublist]
	
	#make list for chisquare outputs
	chi_stats = 4*[[]]

	for taste in sorted(grouped_df.taste.unique()):
		ax = axes_list.pop(0)
        #Separate conditional data
		query_1 = grouped_df.query('taste == @taste and band ==@band and pval_cat == 0 and Condition == @conditions[0]')
		query_2 = grouped_df.query('taste == @taste and band ==@band and pval_cat == 0 and Condition == @conditions[1]')
        
		#Perform chi-aquare test
		for tbin in range(len(sorted(grouped_df.time_bin.unique()))):
			 chi_table = np.array([[np.array([query_1['time_bin'].value_counts()[x] \
                 if x in query_1['time_bin'].unique() else 0 for x in \
                 sorted(grouped_df.time_bin.unique())])[tbin],len(query_1.groupby(['Animal', 'unit']))-np.array([query_1['time_bin'].value_counts()[x] \
                 if x in query_1['time_bin'].unique() else 0 for x in \
                 sorted(grouped_df.time_bin.unique())])[tbin]],
				[np.array([query_2['time_bin'].value_counts()[x] \
                 if x in query_2['time_bin'].unique() else 0 for x in \
                 sorted(grouped_df.time_bin.unique())])[tbin],len(query_2.groupby(['Animal', 'unit']))-np.array([query_2['time_bin'].value_counts()[x] \
                 if x in query_2['time_bin'].unique() else 0 for x in \
                 sorted(grouped_df.time_bin.unique())])[tbin]]])

			 chi_stats[taste]=chi_stats[taste]+[tuple(chi2_contingency(chi_table)[0:2])]

		#Create vector for sigvalues
		sig_vector =np.array([x[1] for x in chi_stats[taste]]) #find p values
		sig_vector= np.array([x*(x<=0.05) for x in sig_vector]) #convert all non-sig to 0
		sig_vector[sig_vector == 0] = 'nan' #convert to nans for plotting
				
        #Applied a first order gaussian filter to smooth lines
		p1 = ax.plot(sorted(grouped_df.time_bin.unique()),
                gaussian_filter1d(np.array([query_1['time_bin'].value_counts()[x] \
                    if x in query_1['time_bin'].unique() else 0 for x in \
                    sorted(grouped_df.time_bin.unique())])/\
                    len(query_1.groupby(['Animal', 'unit']))*100,sigma=1),
                color = 'black',linewidth = 2)
		
		p2 = ax.plot(sorted(grouped_df.time_bin.unique()),
                gaussian_filter1d(np.array([query_2['time_bin'].value_counts()[x] \
                    if x in query_2['time_bin'].unique() else 0 for x in \
                    sorted(grouped_df.time_bin.unique())])/\
                    len(query_2.groupby(['Animal', 'unit']))*100,sigma=1),
                color = 'red',linewidth = 2)	
        		
		#Find max y value for sig plotting and update sigvector vals
		max_y = np.max([np.array([query_1['time_bin'].value_counts()[x] \
                    if x in query_1['time_bin'].unique() else 0 for x in \
                    sorted(grouped_df.time_bin.unique())])/\
                    len(query_1.groupby(['Animal', 'unit']))*100,np.array([query_2['time_bin'].value_counts()[x] \
                    if x in query_2['time_bin'].unique() else 0 for x in \
                    sorted(grouped_df.time_bin.unique())])/\
                    len(query_2.groupby(['Animal', 'unit']))*100])
				
		sig_vector[sig_vector >= 0] = np.ceil(max_y+1) #convert to nans for plotting
				
		#annotate plot
		p3 = ax.scatter(sorted(grouped_df.time_bin.unique()),sig_vector,c="g", s=80, alpha=0.5, marker=r'$\clubsuit$',
            label="Luck")
		
		ax.set_title(identities[taste],size=15,y=1)
	fig.suptitle([x[0] for x in list(freq_vals.values())][band],size=15,y=1)
	fig.legend(conditions,loc = (0.42, 0), ncol=4)
	fig.subplots_adjust(hspace=0.25,wspace = 0.05)
	plt.suptitle('%s Sig. Spike-Phase Locking\n %s: %i cells,  %s: %i cells' %([x[0] for x in list(freq_vals.values())][band],conditions[0],len(query_1.groupby(['Animal', 'unit'])),conditions[1],len(query_2.groupby(['Animal', 'unit']))),size=18,fontweight="bold")  
	fig.savefig('./Grouped_Phase_lock_analyses/SPL_Distributions/'+'%s_SPL_N_%i_animals.png' %([x[0] for x in list(freq_vals.values())][band], len(grouped_df.Animal.unique())))   
	plt.close(fig)		






