# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd

def create_grid_plots(array, array_name, plot_type = 'line'):

    inds = list(np.ndindex(array.shape[:3]))

    # Create plots
    fig_list = [
            plt.subplots(
                len(unique_lasers), 
                len(tastes),
                sharex=True, sharey=True,
                figsize = (len(tastes)*4, 4*len(unique_lasers))) \
            for i in range(len(channel_names))] 

    # Make sure axes are 2D
    fin_fig_list = []
    for fig,ax in fig_list:
        if len(unique_lasers) == 1:
            ax = ax[np.newaxis,:]
        elif len(tastes) == 1:
            ax = ax[:,np.newaxis]
        fin_fig_list.append((fig,ax))

    sub_inds = sorted(list(set([x[1:] for x in inds])))
    for chan_num, (fig,ax) in enumerate(fin_fig_list):
        fig.suptitle(channel_names[chan_num] + ' : ' + array_name)
        for row,col in sub_inds:
            this_ax = ax[row,col]
            this_dat = array[chan_num, row, col]
            if col == 0:
                this_ax.set_ylabel(f"Laser: {unique_lasers[row]}")
            if row == ax.shape[0]-1:
                this_ax.set_xlabel('Time post-stim (ms)')
            this_ax.set_title(tastes[col])
            if plot_type == 'line':
                this_ax.plot(x[plot_indices], this_dat[plot_indices])
                this_ax.set_ylim([0,1])
            else:
                this_ax.pcolormesh(
                        x[plot_indices],
                        np.arange(this_dat.shape[0]),
                        this_dat[:,plot_indices],
                        shading = 'nearest'
                        )
            this_ax.axvline(0, 
                    color = 'red', linestyle = '--', 
                    linewidth = 2, alpha = 0.7)

    for this_name, this_fig in zip(channel_names, fin_fig_list):
        this_fig[0].savefig(
                os.path.join(
                    plot_dir, 
                    f'{this_name}_{array_name}.png'))
        plt.close(this_fig[0])
    #plt.show()

############################################################

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

os.chdir(dir_name)

file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r')

all_nodes = list(hf5.get_node('/emg_BSA_results')._f_iter_nodes())
channel_names = [x._v_name for x in all_nodes \
        if 'group' in str(x.__class__)]

# Pull the data from the /ancillary_analysis node
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]
gapes=hf5.root.emg_BSA_results.gapes[:]
ltps=hf5.root.emg_BSA_results.ltps[:]
sig_trials=hf5.root.emg_BSA_results.sig_trials[:]
emg_BSA_results=hf5.root.emg_BSA_results.emg_BSA_results_final[:]
# Reading single values from the hdf5 file seems hard, 
# needs the read() method to be called
pre_stim=hf5.root.ancillary_analysis.pre_stim.read()
pre_stim = int(pre_stim)
# Also maintain a counter of the number of trials in the analysis

# Close the hdf5 file
hf5.close()

plot_dir = os.path.join(
        dir_name,
        'emg_output',
        'BSA_plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))
with open(json_path[0], 'r') as params_file:
    info_dict = json.load(params_file)

params_path = glob.glob(os.path.join(dir_name, dir_basename + '.params'))
with open(params_path[0], 'r') as params_file:
    params_dict = json.load(params_file)
                
time_limits = [int(x) for x in params_dict['psth_params']['durations']]
# TODO: Fix PSTH start time to be negative if pre-stim
# Due to idiosyncracy in convention, the pre-stim time needs to be reversed
time_limits[0]*=-1

## Ask the user for the time limits to plot the results upto
#time_limits = easygui.multenterbox(
#    msg = 'Enter the time limits to be used in the plots', 
#    fields = ['Pre stim (ms)', 'Post stim (ms)'])
#for i in range(len(time_limits)):
#    time_limits[i] = int(time_limits[i])

# Get an array of x values to plot the average probability of 
# gaping or licking across time
x = np.arange(gapes.shape[-1]) - pre_stim

# Get the indices of x that need to be plotted based on the chosen time limits
plot_indices = np.where((x >= time_limits[0])*(x <= time_limits[1]))[0]

tastes = info_dict['taste_params']['tastes']
# Ask the user for the names of the tastes in the dataset
#tastes = easygui.multenterbox(
#        msg = 'Enter the names of the tastes used in the experiments', 
#        fields = ['Taste{:d}'.format(i+1) for i in range(gapes.shape[1])])

mean_gapes = gapes.mean(axis=-2)
mean_ltps = ltps.mean(axis=-2)


# TODO: If trials are uneven, mean plots aren't made for tastes with
#       with fewer trials, likely due to average of NaNs. Needs fixing
# Generate grid plots
create_grid_plots(mean_gapes, 'mean_gapes')
create_grid_plots(mean_ltps, 'mean_ltps')

create_grid_plots(gapes, 'gapes', plot_type = 'im')
create_grid_plots(ltps, 'ltps', plot_type = 'im')

# TODO: Generate overlay plots

