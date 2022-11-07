"""
Changes to be made:
    1) Add flag for throwing out waveforms at blech_clust step
    2) Convert pred_spikes + pred_noice plots to datashader
    3) Number of noise clusters = 7
    4) Number of spikes clusters = 3
    5) Save predictions so they can be used in UMAP plot

Model monitoring:
    1) Monitoring input and output data distributions

Steps:
    1) Load Data
    2) Preprocess
        a) Bandpass filter
        b) Extract and dejitter spikes
        c) Extract amplitude (accounting for polarity) 
        d) Extract energy and scale waveforms by energy
        e) Perform PCA
        f) Scale all features using StandardScaler
    3) Perform clustering
"""

############################################################
#|_ _|_ __ ___  _ __   ___  _ __| |_ ___  
# | || '_ ` _ \| '_ \ / _ \| '__| __/ __| 
# | || | | | | | |_) | (_) | |  | |_\__ \ 
#|___|_| |_| |_| .__/ \___/|_|   \__|___/ 
#              |_|                        
############################################################

from sklearn.mixture import GaussianMixture as gmm
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pylab as plt
import json
import sys
import numpy as np
import tables
import os
import shutil
import matplotlib
matplotlib.use('Agg')
from joblib import load
import subprocess

## Import 3rd party code
from utils import blech_waveforms_datashader
from utils import memory_monitor as mm
from utils.clustering import *

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

############################################################
# Setting up model

home_dir = os.environ.get("HOME")
model_dir = f'{home_dir}/Desktop/neuRecommend/model'
# Run download model script to make sure latest model is being used
process = subprocess.Popen(f'python {home_dir}/Desktop/blech_clust/utils/download_wav_classifier.py', shell=True)
# Forces process to complete before proceeding
stdout, stderr = process.communicate()
# If model_dir still doesn't exist, then throw an error
if not os.path.exists(model_dir):
    raise Exception("Couldn't download model")
    

pred_pipeline_path = f'{model_dir}/xgboost_full_pipeline.dump'
feature_pipeline_path = f'{model_dir}/feature_engineering_pipeline.dump'

sys.path.append(f'{home_dir}/Desktop/neuRecommend/src/create_pipeline')
from feature_engineering_pipeline import *

clf_threshold_path = f'{model_dir}/proba_threshold.json'
with open(clf_threshold_path,'r') as this_file:
    out_dict = json.load(this_file)
clf_threshold = out_dict['threshold']

############################################################
#|  ___|   _ _ __   ___ ___ 
#| |_ | | | | '_ \ / __/ __|
#|  _|| |_| | | | | (__\__ \
#|_|   \__,_|_| |_|\___|___/
############################################################

def ifisdir_rmdir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


def gen_window_plots(
        filt_el,
        window_len,
        window_count,
        sampling_rate,
        spike_times,
        mean_val,
        threshold,
                ):
    windows_in_data = len(filt_el) // (window_len * sampling_rate)
    window_markers = np.linspace(0,
                                 int(windows_in_data*(window_len * sampling_rate)),
                                 int(windows_in_data))
    window_markers = np.array([int(x) for x in window_markers])
    chosen_window_inds = np.vectorize(np.int)(np.sort(np.random.choice(
        np.arange(windows_in_data), window_count)))
    chosen_window_markers = [(window_markers[x-1], window_markers[x])
                             for x in chosen_window_inds]
    chosen_windows = [filt_el[start:end]
                      for (start, end) in chosen_window_markers]
    # For each window, extract detected spikes
    chosen_window_spikes = [np.array(spike_times)
                            [(spike_times > start)*(spike_times < end)] - start
                            for (start, end) in chosen_window_markers]

    fig, ax = plt.subplots(len(chosen_windows), 1,
                           sharex=True, sharey=True, figsize=(10, 10))
    for dat, spikes, this_ax in zip(chosen_windows, chosen_window_spikes, ax):
        this_ax.plot(dat, linewidth=0.5)
        this_ax.hlines(mean_val + threshold, 0, len(dat))
        this_ax.hlines(mean_val - threshold, 0, len(dat))
        if len(spikes) > 0:
            this_ax.scatter(spikes, np.repeat(
                mean_val, len(spikes)), s=5, c='red')
        this_ax.set_ylim((mean_val - 1.5*threshold,
                          mean_val + 1.5*threshold))
    return fig

def gen_datashader_plot(
        slices_dejittered,
        cluster_points,
        x,
        threshold,
        electrode_num,
        sampling_rate,
        cluster,
        ):
    fig, ax = \
        blech_waveforms_datashader.waveforms_datashader(
            slices_dejittered[cluster_points, :],
            x,
            downsample=False,
            threshold=threshold,
            dir_name="datashader_temp_el" + str(electrode_num))

    ax.set_xlabel('Sample ({:d} samples per ms)'.
                  format(int(sampling_rate/1000)))
    ax.set_ylabel('Voltage (microvolts)')
    ax.set_title('Cluster%i' % cluster)
    return fig, ax

def gen_isi_hist(
        times_dejittered,
        cluster_points,
        ):
    fig = plt.figure()
    cluster_times = times_dejittered[cluster_points]
    ISIs = np.ediff1d(np.sort(cluster_times))
    ISIs = ISIs/30.0
    max_ISI_val = 20
    bin_count = 100
    neg_pos_ISI = np.concatenate((-1*ISIs, ISIs), axis=-1)
    hist_obj = plt.hist(
        neg_pos_ISI,
        bins=np.linspace(-max_ISI_val, max_ISI_val, bin_count))
    plt.xlim([-max_ISI_val, max_ISI_val])
    # Scale y-lims by all but the last value
    upper_lim = np.max(hist_obj[0][:-1])
    if upper_lim:
        plt.ylim([0, upper_lim])
    plt.title("2ms ISI violations = %.1f percent (%i/%i)"
              % ((float(len(np.where(ISIs < 2.0)[0])) /
                  float(len(cluster_times)))*100.0,
                 len(np.where(ISIs < 2.0)[0]),
                 len(cluster_times)) + '\n' +
              "1ms ISI violations = %.1f percent (%i/%i)"
              % ((float(len(np.where(ISIs < 1.0)[0])) /
                  float(len(cluster_times)))*100.0,
                 len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
    return fig

def remove_too_large_waveforms(
        cluster_points,
        amplitudes,
        wf_amplitude_sd_cutoff
        ):
    this_cluster = predictions[cluster_points]
    cluster_amplitudes = amplitudes[cluster_points]
    cluster_amplitude_mean = np.mean(cluster_amplitudes)
    cluster_amplitude_sd = np.std(cluster_amplitudes)
    reject_wf = np.where(cluster_amplitudes <= cluster_amplitude_mean
                         - wf_amplitude_sd_cutoff*cluster_amplitude_sd)[0]
    this_cluster[reject_wf] = -1
    return this_cluster

############################################################
# | |    ___   __ _  __| |
# | |   / _ \ / _` |/ _` |
# | |__| (_) | (_| | (_| |
# |_____\___/ \__,_|\__,_|
############################################################


# Read blech.dir, and cd to that directory
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
    dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

electrode_num = int(sys.argv[1])

# Check if the directories for this electrode number exist -
# if they do, delete them (existence of the directories indicates a
# job restart on the cluster, so restart afresh)
dir_list = [f'./Plots/{electrode_num:02}',
            f'./spike_waveforms/electrode{electrode_num:02}',
            f'./spike_times/electrode{electrode_num:02}',
            f'./clustering_results/electrode{electrode_num:02}']
for this_dir in dir_list:
    ifisdir_rmdir(this_dir)
    os.makedirs(this_dir)
base_plot_dir = dir_list[0]

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
    if files[-6:] == 'params':
        params_file = files

with open(params_file, 'r') as params_file_connect:
    params_dict = json.load(params_file_connect)

# Ideally one would access the params_dict and not have to define variables
# But one step at a time
for key, value in params_dict.items():
    globals()[key] = value

# Open up hdf5 file, and load this electrode number
hf5 = tables.open_file(hdf5_name, 'r')
raw_el = hf5.get_node(f'/raw/electrode{electrode_num:02}')[:]
#exec(f"raw_el = hf5.root.raw.electrode{electrode_num:02}[:]")
hf5.close()

############################################################
# High bandpass filter the raw electrode recordings
filt_el = get_filtered_electrode(
    raw_el,
    freq=[bandpass_lower_cutoff,
          bandpass_upper_cutoff],
    sampling_rate=sampling_rate)

# Delete raw electrode recording from memory
del raw_el

############################################################
# Calculate the 3 voltage parameters
breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
                    * int(sampling_rate))/len(filt_el)
test_el = np.reshape(filt_el[:int(sampling_rate)
                             * int(len(filt_el)/sampling_rate)],
                     (-1, int(sampling_rate)))
breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0])
                    for i in range(len(test_el))]
breaches_per_sec = np.array(breaches_per_sec)
secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
if secs_above_cutoff == 0:
    mean_breach_rate_persec = 0
else:
    mean_breach_rate_persec = np.mean(breaches_per_sec[
        np.where(breaches_per_sec > 0)[0]])

# And if they all exceed the cutoffs,
# assume that the headstage fell off mid-experiment
recording_cutoff = int(len(filt_el)/sampling_rate)
if breach_rate >= max_breach_rate and \
        secs_above_cutoff >= max_secs_above_cutoff and \
        mean_breach_rate_persec >= max_mean_breach_rate_persec:
    # Find the first 1 second epoch where the number of cutoff breaches
    # is higher than the maximum allowed mean breach rate
    recording_cutoff = np.where(breaches_per_sec >
                                max_mean_breach_rate_persec)[0][0]

# Dump a plot showing where the recording was cut off at
fig = plt.figure()
plt.plot(np.arange(test_el.shape[0]), np.mean(test_el, axis=1))
plt.plot((recording_cutoff, recording_cutoff),
         (np.min(np.mean(test_el, axis=1)),
          np.max(np.mean(test_el, axis=1))), 'k-', linewidth=4.0)
plt.xlabel('Recording time (secs)')
plt.ylabel('Average voltage recorded per sec (microvolts)')
plt.title('Recording cutoff time (indicated by the black horizontal line)')
fig.savefig(os.path.join(base_plot_dir, 'cutoff_time.png'), bbox_inches='tight')
plt.close("all")

#############################################################
# | __ )  ___  __ _(_)_ __    _ __  _ __ ___   ___ ___  ___ ___
# |  _ \ / _ \/ _` | | '_ \  | '_ \| '__/ _ \ / __/ _ \/ __/ __|
# | |_) |  __/ (_| | | | | | | |_) | | | (_) | (_|  __/\__ \__ \
# |____/ \___|\__, |_|_| |_| | .__/|_|  \___/ \___\___||___/___/
#            |___/          |_|
#############################################################

# Then cut the recording accordingly
filt_el = filt_el[:recording_cutoff*int(sampling_rate)]

slices, spike_times, polarity, mean_val, threshold = \
    extract_waveforms_abu(filt_el,
                          spike_snapshot=[spike_snapshot_before,
                                          spike_snapshot_after],
                          sampling_rate=sampling_rate)

############################################################
# Extract windows from filt_el and plot with threshold overlayed
window_len = 0.2  # sec
window_count = 10
fig = gen_window_plots(
    filt_el,
    window_len,
    window_count,
    sampling_rate,
    spike_times,
    mean_val,
    threshold,
)
fig.savefig(os.path.join(base_plot_dir, 'bandapass_trace_snippets.png'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
############################################################

# Delete filtered electrode from memory
del filt_el, test_el

# Dejitter these spike waveforms, and get their maximum amplitudes
# Slices are returned sorted by amplitude polaity
slices_dejittered, times_dejittered = \
    dejitter_abu3(slices,
                  spike_times,
                  polarity=polarity,
                  spike_snapshot=[spike_snapshot_before, spike_snapshot_after],
                  sampling_rate=sampling_rate)

############################################################
# Load full pipeline and perform prediction on slices_dejittered
feature_pipeline = load(feature_pipeline_path)
pred_pipeline = load(pred_pipeline_path)

#clf_pred = pred_pipeline.predict(slices_dejittered)
clf_prob = pred_pipeline.predict_proba(slices_dejittered)[:,1]
clf_pred = clf_prob >= clf_threshold

#fig,ax = plt.subplots(1,2, figsize = (10,5))
fig = plt.figure(figsize = (10,5))
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,2,4)
spike_dat = slices_dejittered[clf_pred==1]
spike_times = times_dejittered[clf_pred==1] 
spike_prob = clf_prob[clf_pred==1]
x = np.arange(spike_dat.shape[1])
ax0.plot(x, spike_dat[::10].T, c = 'k', alpha = 0.1)
ax1.scatter(spike_times, spike_prob, s = 1) 
ax1.set_ylabel('Spike probability')
ax2.hist(spike_times, bins = 50)
ax2.set_ylabel('Binned Counts')
ax2.set_xlabel('Time')
fig.suptitle('Predicted Spike Waveforms' + '\n' + f'Count : {spike_dat.shape[0]}')
fig.savefig(os.path.join(base_plot_dir, f'{electrode_num}_pred_spikes.png'),
            bbox_inches='tight')
plt.close(fig)

# Pull out noise info
noise_slices = slices_dejittered[clf_pred==0]
noise_times = times_dejittered[clf_pred==0]
noise_prob = clf_prob[clf_pred==0]


# Cluster noise and plot waveforms + times on single plot
dat_thresh = 10000
zscore_noise_slices = zscore(noise_slices, axis=-1)
noise_train_set = zscore_noise_slices[np.random.choice(np.arange(noise_slices.shape[0]),
                                  int(np.min((noise_slices.shape[0], dat_thresh))))]
noise_pca_obj = PCA(n_components=1).fit(noise_train_set)
noise_pca = noise_pca_obj.transform(zscore_noise_slices)
# Don't need multiple restarts, this is just for visualization, not actual clustering
model = gmm(
    n_components=5,
    max_iter=num_iter,
    n_init=1,
    tol=thresh).fit(noise_train_set)

predictions = model.predict(zscore_noise_slices)

clust_num = len(np.unique(predictions))
#fig = plt.figure(figsize = (20,10))
#wav_ax_list = [fig.add_subplot(clust_num, 2, (2*i)+1) for i in range(clust_num)]
#times_ax = fig.add_subplot(1,2,2)
fig,ax = plt.subplots(clust_num, 2, figsize = (20,10), sharex='col')
ax[0,0].set_title('Waveforms')
ax[0,1].set_title('Spike Times')
plot_max = 500 # Plot at most this many waveforms
#for num, this_ax in enumerate(wav_ax_list):
for num in range(clust_num):
    this_dat = zscore_noise_slices[predictions==num]
    inds = np.random.choice(
            np.arange(this_dat.shape[0]),
            int(np.min((
                this_dat.shape[0],
                plot_max
                )))
            )
    this_dat = this_dat[inds]
    ax[num,0].plot(this_dat.T, color = 'k', alpha = 0.1)
    ax[num,0].set_ylabel(f'Clust {num}')
    this_times = noise_times[predictions==num]
    this_pca = noise_pca[predictions==num]
    #this_prob = noise_prob[predictions==num]
    #ax[num,1].scatter(this_times, this_pca, label = str(num),
    #        alpha = 0.1)
    ax[num,1].hist(this_times, bins = 100)
fig.suptitle('Predicted Noise Waveforms' + '\n' + f'Count : {noise_slices.shape[0]}')
fig.savefig(os.path.join(base_plot_dir, f'{electrode_num}_pred_noise.png'),
            bbox_inches='tight')
plt.close(fig)
#plt.show()

throw_out_noise = True
if throw_out_noise:
    # Remaining data is now only spikes
    slices_dejittered = slices_dejittered[clf_pred==1]
    times_dejittered = times_dejittered[clf_pred==1]
    clf_prob = clf_prob[clf_pred==1]

############################################################

spike_order = np.argsort(times_dejittered)
times_dejittered = times_dejittered[spike_order]
slices_dejittered = slices_dejittered[spike_order]
polarity = polarity[spike_order]

amplitudes = np.zeros((slices_dejittered.shape[0]))
amplitudes[polarity < 0] = np.min(slices_dejittered[polarity < 0], axis=1)
amplitudes[polarity > 0] = np.max(slices_dejittered[polarity > 0], axis=1)

# Delete the original slices and times now that dejittering is complete
del slices
# Save spiketimes for feature timeseries plots
#del spike_times

# Scale the dejittered slices by the energy of the waveforms
scaled_slices, energy = scale_waveforms(slices_dejittered)

# Run PCA on the scaled waveforms
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

# Save the pca_slices, energy and amplitudes to the
# spike_waveforms folder for this electrode
# Save slices/spike waveforms and their times to their respective folders
to_be_saved = ['slices_dejittered', 'times_dejittered',
               'pca_slices', 'energy', 'amplitudes']

this_waveform_path = f'./spike_waveforms/electrode{electrode_num:02}'
save_paths = \
    [f'{this_waveform_path}/spike_waveforms.npy',
     f'./spike_times/electrode{electrode_num:02}/spike_times.npy',
     f'{this_waveform_path}/pca_waveforms.npy',
     f'{this_waveform_path}/energy.npy',
     f'{this_waveform_path}/spike_amplitudes.npy']

for key, path in zip(to_be_saved, save_paths):
    np.save(path, globals()[key])

# Create file for saving plots, and plot explained variance ratios of the PCA
fig = plt.figure()
x = np.arange(len(explained_variance_ratio))
plt.plot(x, explained_variance_ratio, 'x')
plt.title('Variance ratios explained by PCs')
plt.xlabel('PC #')
plt.ylabel('Explained variance ratio')
fig.savefig(os.path.join(base_plot_dir, 'pca_variance.png'),
            bbox_inches='tight')
plt.close("all")


# Make an array of the data to be used for clustering,
# and delete pca_slices, scaled_slices, energy and amplitudes

n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 2))
data[:, 2:] = pca_slices[:, :n_pc]
data[:, 0] = energy[:]/np.max(energy)
data[:, 1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))
data = np.concatenate([data, clf_prob[:,np.newaxis]],axis=-1)

data_labels = [*[f'pc{x}' for x in range(n_pc)],
               'energy',
               'amplitude']

# Standardize features in the data since they
# occupy very uneven scales
standard_data = scaler().fit_transform(data)

del pca_slices
del scaled_slices
del energy

# Set a threshold on how many datapoints are used to FIT the gmm
dat_thresh = 10e3
# Run GMM, from 2 to max_clusters
for i in range(max_clusters-1):
    # If dataset is very large, take subsample for fitting
    train_set = data[np.random.choice(np.arange(data.shape[0]),
                                      int(np.min((data.shape[0], dat_thresh))))]
    model = gmm(
        n_components=i+2,
        max_iter=num_iter,
        n_init=num_restarts,
        tol=thresh).fit(train_set)

    predictions = model.predict(data)

    # Sometimes large amplitude noise waveforms cluster with the
    # spike waveforms because the amplitude has been factored out of
    # the scaled slices.
    # Run through the clusters and find the waveforms that are more than
    # wf_amplitude_sd_cutoff larger than the cluster mean.
    # Set predictions = -1 at these points so that they aren't
    # picked up by blech_post_process
    for cluster in range(i+2):
        cluster_points = np.where(predictions[:] == cluster)[0]
        this_cluster = remove_too_large_waveforms(
                                cluster_points,
                                amplitudes,
                                wf_amplitude_sd_cutoff)
        predictions[cluster_points] = this_cluster

    # Make folder for results of i+2 clusters, and store results there
    clustering_results_dir = f'./clustering_results/electrode{electrode_num:02}/clusters{i+2}'
    ifisdir_rmdir(clustering_results_dir)
    os.mkdir(clustering_results_dir)
    np.save(os.path.join(clustering_results_dir, 'predictions.npy'), predictions)

    # Create file, and plot spike waveforms for the different clusters.
    # Plot 10 times downsampled dejittered/smoothed waveforms.
    # Additionally plot the ISI distribution of each cluster
    this_plot_dir = os.path.join(base_plot_dir, f'{i+2}_clusters_waveforms_ISIs')
    ifisdir_rmdir(this_plot_dir)
    os.mkdir(this_plot_dir)

    x = np.arange(len(slices_dejittered[0])) + 1
    for cluster in range(i+2):
        cluster_points = np.where(predictions[:] == cluster)[0]

        if len(cluster_points) > 0:
            # downsample = False, Prevents waveforms_datashader
            # from FURTHER downsampling the given waveforms for plotting
            # Because in the previous version they were upsampled for clustering

            # Create waveform datashader plot
            fig,ax = gen_datashader_plot(
                        slices_dejittered,
                        cluster_points,
                        x,
                        threshold,
                        electrode_num,
                        sampling_rate,
                        cluster,
                    )
            fig.savefig(os.path.join(this_plot_dir,f'Cluster{cluster}_waveforms'))
            plt.close(fig)

            # Create ISI distribution plot
            fig = gen_isi_hist(
                        times_dejittered,
                        cluster_points,
                    )
            fig.savefig(os.path.join(
                this_plot_dir,f'Cluster{cluster}_ISIs'))
            plt.close("all")

            # Create features timeseries plot
            # And plot histogram of spiketimes
            this_standard_data = standard_data[cluster_points]
            this_spiketimes = spike_times[cluster_points]
            fig,ax = plt.subplots(this_standard_data.shape[1] + 1, 1,
                    figsize = (7,9), sharex=True)
            for this_label, this_dat, this_ax in \
                    zip(data_labels, this_standard_data.T, ax[:-1]):
                this_ax.scatter(this_spiketimes, this_dat, s=0.5, alpha = 0.5)
                this_ax.set_ylabel(this_label)
            ax[-1].hist(this_spiketimes, bins = 50)
            ax[-1].set_ylabel('Spiketime' + '\n' + 'Histogram')
            fig.savefig(os.path.join(
                this_plot_dir,f'Cluster{cluster}_features'))
            plt.close(fig)

        else:
            file_path = os.path.join(this_plot_dir,f'no_spikes_Cluster{cluster}')
            with open(file_path, 'w') as file_connect:
                file_connect.write('')

# Make file for dumping info about memory usage
f = open(f'./memory_monitor_clustering/{electrode_num:02}.txt', 'w')
print(mm.memory_usage_resource(), file=f)
f.close()
print(f'Electrode {electrode_num} complete.')
