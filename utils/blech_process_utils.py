
import utils.clustering as clust
import subprocess
from joblib import load
from sklearn.mixture import GaussianMixture as gmm
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA
from utils import blech_waveforms_datashader
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

############################################################
# Define Functions
############################################################


class cluster_handler():
    """
    Class to handle clustering steps
    """
    def __init__(self, params_dict, data_dir):
        self.params_dict = params_dict
        self.dat_thresh= 10e3
        self.data_dir = data_dir
        self.create_output_dir()

    def return_training_set(self, data):
        """
        Return training set for clustering
        """
        # Get training set
        train_set = data[np.random.choice(np.arange(data.shape[0]),
                                          int(np.min((data.shape[0], self.dat_thresh))))]
        return train_set

    def fit_model(self, train_set, clusters):
        """
        Cluster waveforms
        """
        model= gmm(
            n_components=clusters,
            max_iter=self.params_dict['num_iter'],
            n_init=self.params_dict['num_restarts'],
            tol=self.params_dict['thresh']).fit(train_set)
        return model

    def get_cluster_labels(self, data, model):
        """
        Get cluster labels
        """
        return model.predict(data)

    def save_cluster_labels(self, predictions, electrode_num):
        np.save(os.path.join(clust_results_dir, 'predictions.npy'), predictions)

    def perform_prediction(self):
        self.fit_model(...)
        self.get_cluster_labels(...)

    def clear_large_waveforms(self):
        """
        Clear large waveforms
        """
        # Sometimes large amplitude noise waveforms cluster with the
        # spike waveforms because the amplitude has been factored out of
        # the scaled slices.
        # Run through the clusters and find the waveforms that are more than
        # wf_amplitude_sd_cutoff larger than the cluster mean.
        # Set predictions = -1 at these points so that they aren't
        # picked up by blech_post_process
        for cluster in range(i+2):
            cluster_points= np.where(predictions[:] == cluster)[0]
            this_cluster= bpu.remove_too_large_waveforms(
                cluster_points,
                amplitudes,
                wf_amplitude_sd_cutoff)
            predictions[cluster_points]= this_cluster

    def create_output_dir(self):
        # Make folder for results of i+2 clusters, and store results there
        clust_results_dir= f'./clustering_results/electrode{electrode_num:02}/clusters{i+2}'
        os.mkdir(clust_results_dir)
        clust_plot_dir= f'./Plots/{electrode_num:02}/{i+2}_clusters_waveforms_ISIs'
        os.mkdir(clust_plot_dir)

    def create_output_plots(self):
        # Create file, and plot spike waveforms for the different clusters.
        # Plot 10 times downsampled dejittered/smoothed waveforms.
        # Additionally plot the ISI distribution of each cluster
        x= np.arange(len(slices_dejittered[0])) + 1
        for cluster in range(i+2):
            cluster_points= np.where(predictions[:] == cluster)[0]

            if len(cluster_points) > 0:
                # downsample = False, Prevents waveforms_datashader
                # from FURTHER downsampling the given waveforms for plotting
                # Because in the previous version they were upsampled for clustering

                # Create waveform datashader plot
                fig, ax= bpu.gen_datashader_plot(
                    slices_dejittered,
                    cluster_points,
                    x,
                    threshold,
                    electrode_num,
                    sampling_rate,
                    cluster,
                )
                fig.savefig(os.path.join(
                    clust_plot_dir, f'Cluster{cluster}_waveforms'))
                plt.close("all")

                # Create ISI distribution plot
                fig= bpu.gen_isi_hist(
                    times_dejittered,
                    cluster_points,
                )
                fig.savefig(os.path.join(
                    clust_plot_dir, f'Cluster{cluster}_ISIs'))
                plt.close("all")

                # Create features timeseries plot
                # And plot histogram of spiketimes
                this_standard_data= standard_data[cluster_points]
                this_spiketimes= times_dejittered[cluster_points]
                fig, ax= plt.subplots(this_standard_data.shape[1] + 1, 1,
                                       figsize=(7, 9), sharex=True)
                for this_label, this_dat, this_ax in \
                        zip(feature_labels, this_standard_data.T, ax[:-1]):
                    this_ax.scatter(this_spiketimes, this_dat,
                                    s=0.5, alpha=0.5)
                    this_ax.set_ylabel(this_label)
                ax[-1].hist(this_spiketimes, bins=50)
                ax[-1].set_ylabel('Spiketime' + '\n' + 'Histogram')
                fig.savefig(os.path.join(
                    clust_plot_dir, f'Cluster{cluster}_features'))
                plt.close(fig)

            else:
                file_path= os.path.join(
                    clust_plot_dir, f'no_spikes_Cluster{cluster}')
                with open(file_path, 'w') as file_connect:
                    file_connect.write('')


class classifier_handler():
    """
    Class to handler classifier steps
    """

    def __init__(
            self,
    ):
        home_dir = os.environ.get("HOME")
        model_dir = f'{home_dir}/Desktop/neuRecommend/model'
        # Run download model script to make sure latest model is being used
        process = subprocess.Popen(
            f'python {home_dir}/Desktop/blech_clust/utils/download_wav_classifier.py', shell=True)
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
        with open(clf_threshold_path, 'r') as this_file:
            out_dict = json.load(this_file)
        clf_threshold = out_dict['threshold']

        self.pred_pipeline_path = pred_pipeline_path
        self.feature_pipeline_path = feature_pipeline_path
        self.clf_threshold = clf_threshold
        self.load_pipelines()

    def load_pipelines(self):
        """
        Load feature and prediction pipelines
        """
        self.feature_pipeline = load(feature_pipeline_path)
        self.feature_labels = ...
        self.pred_pipeline = load(pred_pipeline_path)

    def classify_waveforms(self, slices, spiketimes,):
        """
        Classify waveforms
        """
        # Get the probability of each slice being a spike
        clf_prob = self.pred_pipeline.predict_proba(slices)[:, 1]
        clf_pred = clf_prob >= self.clf_threshold
        pred_spike = slices_dejittered[clf_pred == 1]
        pos_spike_times = times_dejittered[clf_pred == 1]
        spike_prob = clf_prob[clf_pred == 1]

        # Pull out noise info
        pred_noise = slices_dejittered[clf_pred == 0]
        noise_times = times_dejittered[clf_pred == 0]
        noise_prob = clf_prob[clf_pred == 0]

    def return_only_spikes(self):
        """
        """
        slices_dejittered= slices_dejittered[clf_pred == 1]
        times_dejittered= times_dejittered[clf_pred == 1]
        clf_prob= clf_prob[clf_pred == 1]
        return slices_dejittered, times_dejittered, clf_prob

    def gen_plots(self):
        fig= plt.figure(figsize=(10, 5))
        ax0= fig.add_subplot(1, 2, 1)
        ax1= fig.add_subplot(2, 2, 2)
        ax2= fig.add_subplot(2, 2, 4)
        x= np.arange(spike_dat.shape[1])
        ax0.plot(x, spike_dat[::10].T, c='k', alpha=0.05)
        ax1.scatter(pos_spike_times, spike_prob, s=1)
        ax1.set_ylabel('Spike probability')
        ax2.hist(pos_spike_times, bins=50)
        ax2.set_ylabel('Binned Counts')
        ax2.set_xlabel('Time')
        fig.suptitle('Predicted Spike Waveforms' + '\n' +
                     f'Count : {spike_dat.shape[0]}')
        fig.savefig(os.path.join(base_plot_dir, f'{electrode_num}_pred_spikes.png'),
                    bbox_inches='tight')
        plt.close(fig)

        # Cluster noise and plot waveforms + times on single plot
        dat_thresh= 10000
        zscore_noise_slices= zscore(noise_slices, axis=-1)
        noise_train_set= zscore_noise_slices[
                   np.random.choice(np.arange(noise_slices.shape[0]),
                   int(np.min((noise_slices.shape[0], dat_thresh))))]
        noise_pca_obj= PCA(n_components=1).fit(noise_train_set)
        noise_pca= noise_pca_obj.transform(zscore_noise_slices)
        # Don't need multiple restarts, this is just for visualization, not actual clustering
        model= gmm(
            n_components=5,
            max_iter=num_iter,
            n_init=1,
            tol=thresh).fit(noise_train_set)

        predictions= model.predict(zscore_noise_slices)

        clust_num= len(np.unique(predictions))
        fig, ax= plt.subplots(clust_num, 2, figsize=(20, 10), sharex='col')
        ax[0, 0].set_title('Waveforms')
        ax[0, 1].set_title('Spike Times')
        plot_max= 1000  # Plot at most this many waveforms
        # for num, this_ax in enumerate(wav_ax_list):
        for num in range(clust_num):
            this_dat= zscore_noise_slices[predictions == num]
            inds= np.random.choice(
                np.arange(this_dat.shape[0]),
                int(np.min((
                    this_dat.shape[0],
                    plot_max
                )))
            )
            this_dat= this_dat[inds]
            ax[num, 0].plot(this_dat.T, color='k', alpha=0.01)
            ax[num, 0].set_ylabel(f'Clust {num}')
            this_times= noise_times[predictions == num]
            this_pca= noise_pca[predictions == num]
            # this_prob = noise_prob[predictions==num]
            # ax[num,1].scatter(this_times, this_pca, label = str(num),
            #        alpha = 0.1)
            ax[num, 1].hist(this_times, bins=100)
        fig.suptitle('Predicted Noise Waveforms' + '\n' +
                     f'Count : {noise_slices.shape[0]}')
        fig.savefig(os.path.join(base_plot_dir, f'{electrode_num}_pred_noise.png'),
                    bbox_inches='tight')
        plt.close(fig)
        # plt.show()


class electrode_handler():
    """
    Class to handle electrode data
    """

    def __init__(self, hdf5_path, electrode_num, params_dict):
        self.params_dict = params_dict
        hf5 = tables.open_file(hdf5_path, 'r')
        el_path = f'/raw/electrode{electrode_num:02}'
        if el_path in hf5:
            self.raw_el = hf5.get_node(el_path)[:]
        else:
            raise Exception(f'{el_path} not in HDF5')
        hf5.close()

    def filter_electrode(self):
        self.filt_el = clust.get_filtered_electrode(
            self.raw_el,
            freq=[self.params_dict['bandpass_lower_cutoff'],
                  self.params_dict['bandpass_upper_cutoff']],
            sampling_rate=self.params_dict['sampling_rate'],)
        # Delete raw electrode recording from memory
        del self.raw_el

    def calc_recording_cutoff(self):
        (
            filt_el,
            breach_rate,
            breaches_per_sec,
            secs_above_cutoff,
            mean_breach_rate_persec,
            recording_cutoff
        ) = calc_recording_cutoff(
            self.filt_el,
            self.params_dict['sampling_rate'],
            self.params_dict['voltage_cutoff'],
            self.params_dict['max_breach_rate'],
            self.params_dict['max_secs_above_cutoff'],
            self.params_dict['max_mean_breach_rate_persec'],
        )

        self.filt_el = filt_el
        self.recording_cutoff = recording_cutoff

    def cutoff_electrode(self):
        self.filt_el = self.filt_el[:self.recording_cutoff *
                                    self.params_dict['sampling_rate']]


class spike_handler():
    """
    Class to handler processing of spikes
    """

    def __init__(self, filt_el, params_dict, dir_name):
        self.filt_el = filt_el
        self.params_dict = params_dict
        self.dir_name = dir_name

    def extract_waveforms(self):
        """
        Extract waveforms from filtered electrode
        """
        slices, spike_times, polarity, mean_val, threshold = \
            clust.extract_waveforms_abu(self.filt_el,
                                        spike_snapshot=[self.params_dict['spike_snapshot_before'],
                                                        self.params_dict['spike_snapshot_after']],
                                        sampling_rate=self.params_dict['sampling_rate'],
                                        threshold_mult=self.params_dict['waveform_threshold'])

        self.slices = slices
        self.spike_times = spike_times
        self.polarity = polarity
        self.mean_val = mean_val
        self.threshold = threshold

    def dejitter_spikes(self):
        """
        Dejitter spikes
        """
        slices_dejittered, times_dejittered = \
            clust.dejitter_abu3(
                self.slices,
                self.spike_times,
                polarity=self.polarity,
                spike_snapshot=[self.params_dict['spike_snapshot_before'],
                                self.params_dict['spike_snapshot_after']],
                sampling_rate=self.params_dict['sampling_rate'])

        # Sort data by time
        spike_order = np.argsort(times_dejittered)
        times_dejittered = times_dejittered[spike_order]
        slices_dejittered = slices_dejittered[spike_order]
        polarity = self.polarity[spike_order]

        self.slices_dejittered = slices_dejittered
        self.times_dejittered = times_dejittered
        self.polarity = polarity
        del self.slices
        del self.spike_times

    def extract_amplitudes(self):
        """
        Extract amplitudes from dejittered spikes
        """
        amplitudes = np.zeros((self.slices_dejittered.shape[0]))
        amplitudes[self.polarity < 0] = np.min(
            self.slices_dejittered[self.polarity < 0],
            axis=1)
        amplitudes[self.polarity > 0] = np.max(
            self.slices_dejittered[self.polarity > 0],
            axis=1)
        self.amplitudes = amplitudes

    def pca_slices(self):
        """
        PCA on dejittered spikes
        """
        # Scale the dejittered slices by the energy of the waveforms
        scaled_slices, energy = clust.scale_waveforms(self.slices_dejittered)

        # Run PCA on the scaled waveforms
        pca_slices, explained_variance_ratio = clust.implement_pca(
            scaled_slices)

    def create_pca_plot(self):
        # Create file for saving plots, and plot explained variance ratios of the PCA
        fig= plt.figure()
        x= np.arange(len(explained_variance_ratio))
        plt.plot(x, explained_variance_ratio, 'x')
        plt.title('Variance ratios explained by PCs')
        plt.xlabel('PC #')
        plt.ylabel('Explained variance ratio')
        fig.savefig(f'./Plots/{electrode_num:02}/pca_variance.png',
                    bbox_inches='tight')
        plt.close("all")

    def extract_features(self,
                         feature_transformer,
                         feature_names):
        ...

    def write_out_spike_data(self):
        """
        Save the pca_slices, energy and amplitudes to the
        spike_waveforms folder for this electrode
        Save slices/spike waveforms and their times to their respective folders
        """
        to_be_saved = ['slices_dejittered',
                       'pca_slices',
                       'energy',
                       'amplitudes',
                       'times_dejittered']

        waveform_dir = f'{self.dir_name}/spike_waveforms/electrode{electrode_num:02}'
        spiketime_dir = \
            f'{self.dir_name}/spike_times/electrode{electrode_num: 02}'
        save_paths = \
            [f'{waveform_dir}/spike_waveforms.npy',
             f'{waveform_dir}/pca_waveforms.npy',
             f'{waveform_dir}/energy.npy',
             f'{waveform_dir}/spike_amplitudes.npy',
             f'{spiketime_dir}/spike_times.npy',
             ]

        for key, path in zip(to_be_saved, save_paths):
            np.save(path, globals()[key])


def ifisdir_rmdir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


def adjust_to_sampling_rate(data, sampling_rate):
    """
    Cut data to have integer number of seconds

    data: numpy array
    sampling_rate: int
    """
    return data[:int(sampling_rate)*int(len(data)/sampling_rate)]


def calc_recording_cutoff(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec
):

    breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
                        * int(sampling_rate))/len(filt_el)
    # Cutoff to have integer numbers of seconds
    filt_el = adjust_to_sampling_rate(filt_el, sampling_rate)
    breaches_per_sec = [len(np.where(filt_el[i] > voltage_cutoff)[0])
                        for i in range(len(filt_el))]
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

    return (filt_el, breach_rate, breaches_per_sec, secs_above_cutoff,
            mean_breach_rate_persec, recording_cutoff)


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


def make_cutoff_plot(filt_el, recording_cutoff, electrode_num):
    """
    Makes a plot showing where the recording was cut off at

    filt_el: numpy array
    recording_cutoff: int
    """
    fig = plt.figure()
    plt.plot(np.arange(filt_el.shape[0]), np.mean(filt_el, axis=1))
    plt.plot((recording_cutoff, recording_cutoff),
             (np.min(np.mean(filt_el, axis=1)),
              np.max(np.mean(filt_el, axis=1))), 'k-', linewidth=4.0)
    plt.xlabel('Recording time (secs)')
    plt.ylabel('Average voltage recorded per sec (microvolts)')
    plt.title('Recording cutoff time (indicated by the black horizontal line)')
    fig.savefig(
        f'./Plots/{electrode_num:02}/cutoff_time.png', bbox_inches='tight')
    plt.close("all")


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
            dir_name="Plots/" + "datashader_temp_el" + str(electrode_num))

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
