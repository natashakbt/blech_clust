# Use the results in Li et al. 2016 to get gapes on taste trials

import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt

from detect_peaks import detect_peaks
from QDA_classifier import QDA
sys.path.append('../..')
from utils.blech_utils import imp_metadata


# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
data_dir = metadata_handler.dir_name
os.chdir(data_dir)

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Extract taste dig-ins from experimental info file
info_dict = metadata_handler.info_dict
params_dict = metadata_handler.params_dict
pre_stim, post_stim = params_dict['spike_array_durations']

psth_durs = params_dict['psth_params']['durations']
psth_durs[0] *= -1
psth_inds = [int(x + pre_stim) for x in psth_durs]

############################################################
# Load and Process Data
############################################################
emg_output_dir = os.path.join(data_dir, 'emg_output')
# Get dirs for each emg CAR
dir_list = glob(os.path.join(emg_output_dir, 'emg*'))
dir_list = [x for x in dir_list if os.path.isdir(x)]

# Load the unique laser duration/lag combos and the trials that correspond
# to them from the ancillary analysis node
trials = hf5.root.ancillary_analysis.trials[:]
unique_lasers = hf5.root.ancillary_analysis.laser_combination_d_l[:]

# Get laser durations to find which trials have laser on
#laser_durations = [x.laser_durations[:] for x in hf5.get_node('/','spike_trains')]
#laser_lags = [x.laser_onset_lag[:] for x in hf5.get_node('/','spike_trains')]
#laser_data_tuples = np.stack([np.vstack([x,y]).T for x,y in zip(laser_durations, laser_lags)])
#
#laser_condition_inds =  [(laser_data_tuples == this_laser).all(-1) for this_laser in unique_lasers]
#
for num, dir_name in enumerate(dir_list):
    emg_basename = os.path.basename(dir_name)
    print(f'Processing {emg_basename}')

    if 'emg_env.npy' not in os.listdir(dir_name):
        raise Exception(f'emg_env.py not found for {dir_name}')
        exit()

    os.chdir(dir_name)

    # Load the required emg data (the envelope and sig_trials)
    env = np.load('emg_env.npy')
    num_tastes, num_trials, _ = env.shape
    env = np.vstack(env)
    sig_trials = np.load('sig_trials.npy').flatten()

    # Now arrange these arrays by (laser condition X taste X trials X time)
    env_final = np.empty(
        (
            len(trials),
            num_tastes,
            int(num_trials/len(trials)),
            env.shape[1]
        ),
        dtype=float)

    # Shape : (laser conditions x tastes x trials)
    sig_trials_final = np.empty(
        (
            len(trials),
            num_tastes,
            int(num_trials/len(trials))
        ),
        dtype=int)

    # Also make an array to store the time of first gape on every trial
    first_gape = np.empty(
        (
            len(trials),
            num_tastes,
            int(num_trials/len(trials))
        ),
        dtype=int)

    # Fill up these arrays
    for i in range(len(trials)):
        for j in range(num_tastes):
            env_final[i, j, :, :] = env[trials[i][np.where(
                (trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)))], :]
            sig_trials_final[i, j, :] = sig_trials[trials[i][np.where(
                (trials[i] >= num_trials*j)*(trials[i] < num_trials*(j+1)))]]

    # Make an array to store gapes (with 1s)
    gapes_Li = np.zeros(env_final.shape)

    # Run through the trials and get burst times,
    # intervals and durations. Also check if these bursts are gapes -
    # if they are, put 1s in the gape array
    for i in range(sig_trials_final.shape[0]):
        for j in range(sig_trials_final.shape[1]):

            max_cols = 3
            max_rows = 5
            rows = int(np.ceil(sig_trials_final.shape[2] / max_cols))
            fig_count = int(np.ceil(rows/max_rows))
            fin_rows = int(np.ceil(rows/fig_count))
            outs = [plt.subplots(fin_rows, max_cols, sharex= True, sharey = True) \
                    for i in range(fig_count)]
            for num, this_out in enumerate(outs):
                this_out[0].suptitle(f'Laser Cond: {unique_lasers[i]}, Taste {j}, fig {num}')
            ax = np.concatenate([x[1].flatten() for x in outs])

            for k in range(sig_trials_final.shape[2]):
                # Get peak indices
                peak_ind = detect_peaks(
                    env_final[i, j, k, :],
                    mpd=85,
                    mph=np.mean(
                        env_final[i, :, :, :pre_stim]) +
                    np.std(env_final[i, :, :, :pre_stim]
                           )
                )

                #plt.plot(env_final[i,j,j]);plt.vlines(peak_ind,0,100, color = 'k');plt.show()

                # Get the indices, in the smoothed signal,
                # that are below the mean of the smoothed signal
                below_mean_ind = np.where(env_final[i, j, k, :] <= np.mean(
                    env_final[i, :, :, :pre_stim]))[0]

                # Throw out peaks if they happen in the pre-stim period
                accept_peaks = np.where(peak_ind > pre_stim)[0]
                peak_ind = peak_ind[accept_peaks]

                # Run through the accepted peaks, and append their breadths to durations.
                # There might be peaks too close to the end of the trial -
                # skip those. Append the surviving peaks to final_peak_ind
                durations = []
                final_peak_ind = []
                for peak in peak_ind:
                    try:
                        left_end = np.where(below_mean_ind < peak)[0][-1]
                        right_end = np.where(below_mean_ind > peak)[0][0]
                    except:
                        continue
                    dur = below_mean_ind[right_end]-below_mean_ind[left_end]
                    if dur > 20.0 and dur <= 200.0:
                        durations.append(dur)
                        final_peak_ind.append(peak)
                durations = np.array(durations)
                peak_ind = np.array(final_peak_ind)

                # plt.plot(env_final[i,j,j])
                # plt.vlines(peak_ind,0,100, color = 'k')
                # for this_peak, this_dur in zip(peak_ind, durations):
                #     plt.axvspan(this_peak - this_dur/2,
                #              this_peak + this_dur/2,
                #              alpha = 0.5, color = 'yellow')
                # plt.show()

                # In case there aren't any peaks or just one peak
                # (very unlikely), skip this trial and mark it 0 on sig_trials
                if len(peak_ind) <= 1:
                    sig_trials_final[i, j, k] = 0
                else:
                    # Get inter-burst-intervals for the accepted peaks,
                    # convert to Hz (from ms)
                    intervals = []
                    for peak in range(len(peak_ind)):
                        # For the first peak,
                        # the interval is counted from the second peak
                        if peak == 0:
                            intervals.append(
                                1000.0/(peak_ind[peak+1] - peak_ind[peak]))
                        # For the last peak, the interval is
                        # counted from the second to last peak
                        elif peak == len(peak_ind) - 1:
                            intervals.append(
                                1000.0/(peak_ind[peak] - peak_ind[peak-1]))
                        # For every other peak, take the largest interval
                        else:
                            intervals.append(
                                1000.0/(
                                    np.amax([(peak_ind[peak] - peak_ind[peak-1]),
                                             (peak_ind[peak+1] - peak_ind[peak])])
                                )
                            )
                    intervals = np.array(intervals)

                    # Now run through the intervals and durations of the accepted
                    # movements, and see if they are gapes.
                    # If yes, mark them appropriately in gapes_Li
                    # Do not use the first movement/peak in the trial -
                    # that is usually not a gape
                    for peak in range(len(durations) - 1):
                        gape = QDA(intervals[peak+1], durations[peak+1])
                        if gape and peak_ind[peak+1] - pre_stim <= post_stim:
                            gapes_Li[i, j, k, peak_ind[peak+1]] = 1.0

                    # If there are no gapes on a trial, mark these as 0
                    # on sig_trials_final and 0 on first_gape.
                    # Else put the time of the first gape in first_gape
                    if np.sum(gapes_Li[i, j, k, :]) == 0.0:
                        sig_trials_final[i, j, k] = 0
                        first_gape[i, j, k] = 0
                    else:
                        first_gape[i, j, k] = np.where(
                            gapes_Li[i, j, k, :] > 0.0)[0][0]

                plot_dat = env_final[i, j, k, psth_inds[0]:psth_inds[1]]
                plot_gapes = gapes_Li[i,j,k, psth_inds[0]:psth_inds[1]]
                t = np.arange(*psth_durs)
                ax[k].plot(t,plot_dat, linewidth = 1)
                #ax[k].vlines(peak_ind, plot_dat.min(), plot_dat.max(), 
                #             color = 'k', alpha = 0.5) 
                #plot_peak_inds = peak_ind - pre_stim
                #wanted_inds = np.logical_and(
                #        plot_peak_inds > psth_inds[0],
                #        plot_peak_inds < psth_inds[1]
                #        )
                #plot_peak_inds = plot_peak_inds[wanted_inds]
                #durations = durations[wanted_inds]
                #for this_peak, this_dur in zip(plot_peak_inds, durations):
                #    ax[k].axvspan(this_peak - this_dur/2,
                #             this_peak + this_dur/2,
                #             alpha = 0.5, color = 'yellow')
                ax[k].plot(t, plot_gapes * -env_final.max(axis=None) * 0.2)
                ax[k].set_ylim([-env_final.max(axis=None) * 0.2,
                                env_final.max(axis=None)])
                ax[k].set_ylabel(f'Trial {k}')

            plot_dir = f'emg_output/gape_classifier_plots/overview/{emg_basename}'
            fin_plot_dir = os.path.join(data_dir, plot_dir)
            if not os.path.exists(fin_plot_dir):
                os.makedirs(fin_plot_dir)
            for num, this_out in enumerate(outs):
                savename = f'laser_{unique_lasers[i]}_taste{j}_fig{num}.png'
                this_out[0].savefig(
                        os.path.join(fin_plot_dir, savename),
                        dpi = 300)
                plt.close(this_out[0])
            #plt.show()

    # Save these results to the hdf5 file
    hf5_base_path = '/emg_gape_classifier'
    hf5_save_path = os.path.join(hf5_base_path, emg_basename)

    if hf5_save_path not in hf5:
        hf5.create_group(hf5_base_path, emg_basename, createparents=True)
    try:
        hf5.remove_node(f'{hf5_save_path}/gapes_Li')
        hf5.remove_node(f'{hf5_save_path}/gape_trials_Li')
        hf5.remove_node(f'{hf5_save_path}/first_gape_Li')
    except:
        pass
    hf5.create_array(hf5_save_path, 'gapes_Li', gapes_Li)
    hf5.create_array(hf5_save_path, 'gape_trials_Li', sig_trials_final)
    hf5.create_array(hf5_save_path, 'first_gape_Li', first_gape)
    hf5.flush()

hf5.close()
