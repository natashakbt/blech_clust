#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:45:34 2019

@author: bradly
"""
#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import pandas as pd
import numpy as np # module for low-level scientific computing
import easygui
from datetime import datetime
from matplotlib.patches import Rectangle
from collections import OrderedDict
import matplotlib.pyplot as plt
import string

# =============================================================================
# #Define functions
# =============================================================================
def to_xy_stone(some_dictionary):
    x, y = [], []
    for i,event in some_dictionary.items():
        y.extend([i]*len(event))
        x.extend(event)
    x, y = np.array(x).astype(np.int64), np.array(y)
    return x,y
	
def plot_events_stone(xval,yval,labs):
	#set color palette
	#colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(labs)))
	
	fig, ax = plt.subplots()
	ax.set_xlim([0, np.max(xval)])
	ax.set_ylim([-0.5, np.max(yval)+0.5])
	for i in range(0, len(yval), 2):
		ax.add_patch(Rectangle((xval[i], yval[i]-0.25), xval[i+1], 0.5, facecolor = 'blue'))
	plt.yticks(range(yval.max()+1),labs);

# =============================================================================
# Import/Open HDF5 File and variables
# =============================================================================

#Get name of directory where the data files and xlsx file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the xlsx file in the directory
file_list = os.listdir('./')
excel_name = ''
for files in file_list:
    if files[-4:] == 'xlsx':
        excel_name = files

#Open file
xl_file = pd.ExcelFile(excel_name)

dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}

#Create empty list for time difference storing
time_diff = [];
for i in len(dfs):  #Flip through all sheets
	#Get sheet name
	sheet = next(iter(dfs))
	
	#Locate start times (00:00:0001) <--Scored this way to allow for pandas processing
	sheet_df = dfs[sheet].reset_index()
	start_rows = sheet_df[sheet_df['Time_Start_(min:sec.ms)'] == sheet_df['Time_Start_(min:sec.ms)'].min()]
	
	newStart = []; newEnd =[]; timediff= []
	for n in range(len(sheet_df)):
		#convert and calculate time differences
		raw_start = datetime.strptime(str(sheet_df['Time_Start_(min:sec.ms)'][n]),"%H:%M:%S.%f")
		raw_end = datetime.strptime(str(sheet_df['Time_End'][n]),"%H:%M:%S.%f")
		raw_diff = datetime.strptime(str(sheet_df['Time_End'][n]),"%H:%M:%S.%f")-datetime.strptime(str(sheet_df['Time_Start_(min:sec.ms)'][n]),"%H:%M:%S.%f")
		
		#store time values
		newStart.append((raw_start.minute)*60000+(raw_start.second)*1000+raw_start.microsecond/1000)
		newEnd.append((raw_end.minute)*60000+(raw_end.second)*1000+raw_end.microsecond/1000)
		timediff.append((raw_diff.seconds)*1000+raw_diff.microseconds/1000)
		
	sheet_df['Start_time_ms'] = newStart
	sheet_df['End_time_ms'] = newEnd
	sheet_df['Time_diff_ms'] = timediff
	
	for session in range(len(start_rows)): #Flip through video sessions
		#set session query
		sess_query = sheet_df[start_rows.index[session]:start_rows.index[session+1]]
	
		#Create categorical values for events and recode
		sess_query.Event_1 = pd.Categorical(sess_query.Event_1)
		sess_query['code'] = sess_query.Event_1.cat.codes
	
		#Extract events
		events = sess_query['Event_1'].unique()
		event_codes = sess_query['code'].unique()
		
		event_vals = []
		for event in range(len(sess_query['Event_1'].unique())):
			event_extract = sess_query['Event_1'].unique()[event]
			newQ = sess_query.query('Event_1 == @event_extract')
			
			#Store event times as vectors in list
			event_vals.append(np.array(newQ[['Start_time_ms','End_time_ms']]).flatten())
			
		#Create dictionary using event names as keys
		event_dict = dict(zip(events,event_vals))
		event_dict_coded = dict(zip(event_codes,event_vals))
		beta = OrderedDict(sorted(event_dict_coded.items(), key=lambda x: x[0]))
		alpha = OrderedDict(sorted(event_dict.items(), key=lambda x: x[0]))

		#Plot data
		x,y = to_xy_stone(beta)
		plot_events_stone(x,y,np.array([*alpha]))

# =============================================================================
# 		newT = []
# 		for n in range(len(sess_query)):
# 			raw_t = datetime.strptime(str(sess_query['Time_End'][n]),"%H:%M:%S.%f")-datetime.strptime(str(sess_query['Time_Start_(min:sec.ms)'][n]),"%H:%M:%S.%f")
# 			newT.append((raw_t.seconds)*1000+raw_t.microseconds/1000)
# 			
# 		sess_query['time_diff (ms)'] = newT
# 		#events = sess_query['Event_1'].unique()
# =============================================================================

# =============================================================================
# #Define Variables
# =============================================================================
#define color palette
colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 4))
	
def generate_data(N = 20):
    data = [random.randrange(3) for x in range(N)]
    A = [i for i, x in enumerate(data) if x == 0]
    B = [i for i, x in enumerate(data) if x == 1]
    C = [i for i, x in enumerate(data) if x == 2]
    return A,B,C

def to_xy(*events):
    x, y = [], []
    for i,event in enumerate(events):
        y.extend([i]*len(event))
        x.extend(event)
    x, y = np.array(x), np.array(y)
    return x,y

def event_string(x,y):
    labels = np.array(list(string.ascii_uppercase))        
    seq = labels[y[np.argsort(x)]]
    return seq.tostring()

def plot_events(x,y):
    labels = np.array(list(string.ascii_uppercase))    
    plt.hlines(y, x, x+1, lw = 2, color = 'red')
    plt.ylim(max(y)+0.5, min(y)-0.5)
    plt.yticks(range(y.max()+1), labels)
    plt.show()
	
A,B,C = generate_data(20)
x,y = to_xy(A,B,C)

print(event_string(x,y))
plot_events(x,y)


def plot_events_stone(x,y,labs):
    labels = labs  
    plt.hlines(y, x, x+1, lw = 200, color = 'red')
    plt.ylim(max(y)+0.5, min(y)-0.5)
    plt.yticks(range(y.max()+1), labels)
    plt.show()
	
	

#plot using hbar
xx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
yy = np.array([0, 0, 1, 0, 0, 0, 1, 1, 2, 2, 0, 2, 0, 2])

labels = np.array(list(string.ascii_uppercase))    
plt.barh(yy, [1]*len(xx), left=x, color = 'red', edgecolor = 'red', align='center', height=1)

plt.barh(y, [1]*len(x), left=x, color = 'red', edgecolor = 'red', align='center', height=1)
plt.ylim(max(y)+0.5, min(y)-0.5); 
plt.yticks(range(y.max()+1), np.array(events)); plt.xlim(0,1000)

fig, ax = plt.subplots()
ax.set_xlim([0, np.max(x)])
ax.set_ylim([0, np.max(y)+1])
for i in range(0, len(y), 2):
    ax.add_patch(Rectangle((x[i], y[i]), x[i+1], 1, facecolor = "red"))

plt.yticks(range(y.max()+1), np.array([*alpha]));
  

plt.axvspan(x[0], x[1], facecolor='g', alpha=0.5); plt.axvspan(x[3], x[4], facecolor='g', alpha=0.5);