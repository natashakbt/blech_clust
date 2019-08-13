# ==============================
# Setup
# ==============================

#Import necessary tools
import numpy as np
import tables
import easygui
import os
import glob
import matplotlib.pyplot as plt
import re
from tqdm import trange
#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt

#Get name of directory where the data files and hdf5 file sits, 
#and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the hdf5 file in the directory
hdf5_name = glob.glob('**.h5')[0]

#Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# ==============================
# Select channels to read
# ==============================

#Create vector of electode numbers that have neurons 
#on them (from unit_descriptor table)
#Some electrodes may record from more than one neuron 
#(shown as repeated number in unit_descriptor); 
#Remove these duplicates within array
electrodegroup = np.unique(hf5.root.unit_descriptor[:]['electrode_number'])

## List all appropriate dat files
Raw_Electrodefiles = np.sort(glob.glob('*amp*dat*'))
Raw_Electrodefiles = Raw_Electrodefiles[electrodegroup]

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#======================================
# This section is not needed till we want to process channels
# without any cells on them. Removal of EMG electrodes already
# happens in blech processing.
#=====================================

## Ask user to specify non-electrode channels
#non_electrode_channels = easygui.multchoicebox(msg='Select ANY non-electrode channels',
#                title='Non-electrode channels',
#                choices=[x[4:9] for x in Raw_Electrodefiles])
## If any non-electrode channels present
## convert selected channels to inds for easing processing
#if non_electrode_channels: 
#        non_electrode_channels_inds = [ind for ind in range(len(Raw_Electrodefiles)) \
#                        for noncell_channel in non_electrode_channels \
#                        if re.search(noncell_channel,Raw_Electrodefiles[ind]) \
#                        is not None]
#        # Remove channels which are not electrodes
#        Raw_Electrodefiles = [Raw_Electrodefiles[x] for x in range(len(Raw_Electrodefiles)) \
#                        if x not in non_electrode_channels_inds]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

# ==============================
# Extract Raw Data 
# ==============================

#Specify filtering parameters (linear-phase finite impulse response filter) 
#to define filter specificity across electrodes
Boxes = ['low','high','Sampling Rate']
freqparam = list(map(int,easygui.multenterbox(
    'Specify LFP bandpass filtering paramters and sampling rate',
    'Low-Frequency Cut-off (Hz)', 
    Boxes, [1,300,30000]))) 

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2, 
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate], 
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

#Check if LFP data is already within file and remove node if so. 
#Create new raw LFP group within H5 file. 
try:
    hf5.remove_node('/raw_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'raw_LFP')

#Loop through each neuron-recording electrode (from .dat files), 
#filter data, and create array in new LFP node
for i in trange(len(Raw_Electrodefiles)):

    #Read and filter data
    data = np.fromfile(Raw_Electrodefiles[i], dtype = np.dtype('int16'))
    filt_el = get_filtered_electrode(data = data,
                                    low_pass = freqparam[0],
                                    high_pass = freqparam[1],
                                    sampling_rate = freqparam[2])

    hf5.create_array('/raw_LFP','electrode%i' % electrodegroup[i], filt_el)
    hf5.flush()
    del filt_el, data

# Grab the names of the arrays containing digital inputs, 
# and pull the data into a numpy array
dig_in_nodes = hf5.list_nodes('/digital_in')
dig_in = []
dig_in_pathname = []
for node in dig_in_nodes:
    dig_in_pathname.append(node._v_pathname)
    exec("dig_in.append(hf5.root.digital_in.%s[:])" \
                % dig_in_pathname[-1].split('/')[-1])
dig_in = np.array(dig_in)

# Get the stimulus delivery times - 
# take the end of the stimulus pulse as the time of delivery
dig_on = []
for i in range(len(dig_in)):
    dig_on.append(np.where(dig_in[i,:] == 1)[0])
change_points = []
for on_times in dig_on:
    changes = []
    for j in range(len(on_times) - 1):
        if np.abs(on_times[j] - on_times[j+1]) > 30:
            changes.append(on_times[j])
    try:
        # append the last trial which will be missed by this method
        changes.append(on_times[-1]) 
    except:
        # Continue without appending anything if this port wasn't on at all
        pass 
    change_points.append(changes)

# Show the user the number of trials on each digital 
#input channel, and ask them to confirm
check = easygui.ynbox(
        msg = 'Digital input channels: ' + str(dig_in_pathname) + \
                '\n' + 'No. of trials: ' + str([len(changes) \
                for changes in change_points]), 
                title = 'Check and confirm the number of ' + \
                'trials detected on digital input channels')

# Go ahead only if the user approves by saying yes
if check:
    pass
else:
    print("Well, if you don't agree, blech_clust can't do much!")
    sys.exit()

# ==============================
# Write-Out Extracted LFP 
# ==============================

# Ask the user which digital input channels should be 
# used for slicing out LFP arrays, and convert the channel 
# numbers into integers for pulling stuff out of change_points
dig_in_channels = easygui.multchoicebox(
        msg = 'Which digital input channels should be used ' + \
        'to slice out LFP data trial-wise?', 
        choices = ([path for path in dig_in_pathname]))
dig_in_channel_nums = []
for i in range(len(dig_in_pathname)):
    if dig_in_pathname[i] in dig_in_channels:
        dig_in_channel_nums.append(i)

# Ask the user for the pre and post stimulus durations to 
#be pulled out, and convert to integers
durations = easygui.multenterbox(
        msg = 'What are the signal durations pre and post ' + \
        'stimulus that you want to pull out', 
        fields = ['Pre stimulus (ms)', 'Post stimulus (ms)'],
        values = [2000,5000])
for i in range(len(durations)):
    durations[i] = int(durations[i])

# Grab the names of the arrays containing LFP recordings
lfp_nodes = hf5.list_nodes('/raw_LFP')

# Make the Parsed_LFP node in the hdf5 file if it doesn't exist, else move on
try:
    hf5.remove_node('/Parsed_LFP', recursive = True)
except:
    pass
hf5.create_group('/', 'Parsed_LFP')

# Ask if this analysis is looking at more than 1 trial and/or taste
msg   = "Do you want to create LFPs for more than " + \
        "ONE trial (ie. Do you have several tastes) ?"
trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

# Run through the tastes if user said there are more than 1 trial
if trial_check == "Yes":
    for i in range(len(dig_in_channels)):
            num_electrodes = len(lfp_nodes) 
            num_trials = len(change_points[dig_in_channel_nums[i]])
            this_taste_LFPs = np.zeros((
                    num_electrodes, num_trials, durations[0] + durations[1]))
            for electrode in range(num_electrodes):
                for j in range(len(change_points[dig_in_channel_nums[i]])):
                    this_taste_LFPs[electrode, j, :] =\
                        np.mean(
                            lfp_nodes[electrode]\
                            [change_points[dig_in_channel_nums[i]][j] -\
                            durations[0]*30:change_points[dig_in_channel_nums[i]][j] \
                            + durations[1]*30].reshape((-1, 30)), axis = 1)
            
            print (float(i)/len(dig_in_channels)) #Shows progress   
        
            # Put the LFP data for this taste in hdf5 file under /Parsed_LFP
            hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' \
                        % (dig_in_channel_nums[i]), this_taste_LFPs)
            hf5.flush()
        
if trial_check == "No":
    num_electrodes = len(lfp_nodes) 
    num_trials = len(change_points[dig_in_channel_nums[0]])-1
    this_taste_LFPs = np.zeros(
            (num_electrodes, num_trials, durations[0] + durations[1]))
    for electrode in range(num_electrodes):
        this_taste_LFPs[electrode, 0, :] =\
            np.mean(lfp_nodes[electrode][change_points[dig_in_channel_nums[0]][0] -\
                    durations[0]*30:change_points[dig_in_channel_nums[0]][0] + \
                    durations[1]*30].reshape((-1, 30)), axis = 1)
    
        # Put the LFP data for this session in hdf5 file under /Parsed_LFP
    hf5.create_array('/Parsed_LFP', 'dig_in_%i_LFPs' \
            % (dig_in_channel_nums[0]), this_taste_LFPs)
    hf5.flush()

# Ask people if they want to delete rawLFPs or not, that 
#way we offer the option to run analyses in many different ways. 
#(ie. First half V back half)
msg   = "Do you want to delete the Raw LFP data?"
rawLFPdelete = easygui.buttonbox(msg,choices = ["Yes","No"])
if rawLFPdelete == "Yes":
    #Delete data
    hf5.remove_node('/raw_LFP', recursive = True)
hf5.flush()

# ================================================
# Make plots to visually check quality of channels 
# ================================================

# Code copied from LFP_Spectrogram_Stone.py
# Might need cleanup

# =============================================================================
# #Establish User inputs for Variables
# =============================================================================

#Ask if file needs to be split, if yes, split it
split_response = easygui.indexbox(
        msg='Do you need to split these trials?', 
        title='Split trials', choices=('Yes', 'No'), 
        image=None, default_choice='Yes', cancel_choice='No')

# Ask if this analysis is looking at more than 1 trial and/or taste
msg   = "Do you want to perform LFP analyses for more than ONE trial" \
                    "(ie. Do you have several tastes) ?"
trial_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if trial_check == "Yes":
    total_trials = hf5.root.Parsed_LFP.dig_in_1_LFPs[:].shape[1]
    # Ask about subplotting
    msg   = "Do you want saved outputs for each tastant?"
    subplot_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if trial_check == "No":
    total_trials = 1

dig_in_channels = hf5.list_nodes('/digital_in')
dig_in_LFP_nodes = hf5.list_nodes('/Parsed_LFP')

if split_response == 0:
    trial_split = easygui.multenterbox(
            msg = "Put in the number of trials to parse from each of "\
                    "the LFP arrays (only integers)", 
                    fields = [node._v_name for node in dig_in_LFP_nodes], 
                    values = ['15' for node in dig_in_LFP_nodes])

    #Convert all values to integers
    trial_split = list(map(int,trial_split))
    total_sessions = int(total_trials/int(trial_split[0]))

    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:,0:trial_split[node],:]) \
            for node in range(len(dig_in_LFP_nodes))]
    
else:    
    total_sessions = 1
    trial_split = list(map(int,[total_trials for node in dig_in_LFP_nodes]))
    #Create dictionary of all parsed LFP arrays
    LFP_data = [np.array(dig_in_LFP_nodes[node][:]) \
            for node in range(len(dig_in_LFP_nodes))]
    
#Establish timing parameters
if trial_check == "No":
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Taste array start time (ms)', 
                    'Taste array end time (ms)', 
                    'Sampling Rate (samples per second)', 
                    'Signal Window (ms)', 
                    'Window Overlap (ms; default 90%)'], 
            values = ['0','1200000','1000','1000','900'])
        
    #create timing variables
    pre_stim = 0

else:    
    analysis_params = easygui.multenterbox(
            msg = 'Input analysis paramters:', 
            fields = ['Pre-stimulus signal duration (ms; from set-up)',
                    'Post-stimulus signal duration (ms; from set-up)',
                    'Pre-Taste array start time (ms)', 
                    'Taste array end time (ms)'], 
            values = ['2000','5000','0','2500'])
    
    #create timing variables
    pre_stim = int(durations[0])

    # Ask if this analysis is an average of normalization 
    taste_params = easygui.multenterbox(
            msg = 'Input taste identities:', 
            fields = ['Taste 1 (dig_in_1)', 
                    'Taste 2 (dig_in_2)',
                    'Taste 3 (dig_in_3)',
                    'Taste 4 (dig_in_4)'],
            values = ['NaCl','Sucrose','Citric Acid','QHCl'])


# =============================================================================
# #Channel Check
# =============================================================================
# Make directory to store the LFP trace plots. Delete and remake the directory if it exists
try:
        os.system('rm -r '+'./LFP_channel_check')
except:
        pass
os.mkdir('./LFP_channel_check')

#Check to make sure LFPs are "normal" and allow user to remove any that are not
for taste in range(len(LFP_data)):
        #Set data
        channel_data = np.mean(LFP_data[taste],axis=1).T
        t=np.array(list(range(0,np.size(channel_data,axis=0))))
        
        #Create figure
        fig,axes = plt.subplots(nrows=np.size(channel_data,axis=1), 
                ncols=1,sharex=True, sharey=False,figsize=(12, 8), squeeze=False)
        fig.text(0.5, 0.05, 'Milliseconds', ha='center',fontsize=15)
        axes_list = [item for sublist in axes for item in sublist]
        
        for ax, chan in zip(axes.flatten(),range(np.size(channel_data,axis=1))):
        
                ax = axes_list.pop(0)
                ax.set_yticks([])
                ax.plot(t, channel_data[:,chan])
                h = ax.set_ylabel('Channel %s' %(chan))
                h.set_rotation(0)
                ax.vlines(x=pre_stim, ymin=np.min(channel_data[:,chan]),
                        ymax=np.max(channel_data[:,chan]), linewidth=4, color='r')
                
        fig.subplots_adjust(hspace=0,wspace = -0.15)
        fig.suptitle('Dig in {} - '.format(taste) + \
                '%s - Channel Check: %s' %(taste_params[taste], 
                hdf5_name[0:4])+'\n' + 'Raw LFP Traces; Date: %s' \
                                %(re.findall(r'_(\d{6})', 
                hdf5_name)[0]),size=16,fontweight='bold')
        fig.savefig('./LFP_channel_check/' + hdf5_name[0:4] + \
                '_dig_in{}'.format(taste) + \
                '_ %s_%s' %(re.findall(r'_(\d{6})', hdf5_name)[0],
                    taste_params[taste]) + '_channelcheck.png')   

# ==============================
# Close Out 
# ==============================
print("If you want to compress the file to release disk space, " + \
        "run 'blech_hdf5_repack.py' upon completion.")
hf5.close()

