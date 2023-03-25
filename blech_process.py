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
# Imports
############################################################

from utils.blech_utils import (
    imp_metadata,
)
import utils.blech_process_utils as bpu
from utils import memory_monitor as mm
import subprocess
from joblib import load
from sklearn.mixture import GaussianMixture as gmm
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pylab as plt
import json
import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)


############################################################
# Load Data
############################################################


if __name__ == '__main__':
    # Read blech.dir, and cd to that directory
    home_dir = os.getenv('HOME')
    blech_clust_dir = os.path.join(home_dir, 'Desktop', 'blech_clust')
    f = open(os.path.join(blech_clust_dir, 'blech.dir'), 'r')
    dir_name = []
    for line in f.readlines():
        dir_name.append(line)
    f.close()
    dir_name = dir_name[0][:-1]

    metadata_handler = imp_metadata([[], dir_name])
    os.chdir(metadata_handler.dir_name)

    electrode_num = int(sys.argv[1])
    params_dict = metadata_handler.params_dict

    # Check if the directories for this electrode number exist -
    # if they do, delete them (existence of the directories indicates a
    # job restart on the cluster, so restart afresh)
    dir_list = [f'./Plots/{electrode_num:02}',
                f'./spike_waveforms/electrode{electrode_num:02}',
                f'./spike_times/electrode{electrode_num:02}',
                f'./clustering_results/electrode{electrode_num:02}']
    for this_dir in dir_list:
        bpu.ifisdir_rmdir(this_dir)
        os.makedirs(this_dir)

    ############################################################
    # Preprocessing
    ############################################################
    # Open up hdf5 file, and load this electrode number
    electrode = bpu.electrode_handler(
                      metedata_handler.hdf5_path,
                      electrode_num,
                      params_dict)

    electrode.filter_electrode()

    # Calculate the 3 voltage parameters
    electrode.calc_recording_cutoff()

    # Dump a plot showing where the recording was cut off at
    bpu.make_cutoff_plot(
        electrode.filt_el,
        electrode.recording_cutoff,
        electrode_num)

    # Then cut the recording accordingly
    electrode.cutoff_electrode()

    #############################################################
    # Process Spikes
    #############################################################

    spike_set = bpu.spike_handler(electrode.filt_el, params_dict)
    spike_set.extract_waveforms()

    ############################################################
    # Extract windows from filt_el and plot with threshold overlayed
    window_len= 0.2  # sec
    window_count= 10
    fig= gen_window_plots(
        electrode.filt_el,
        window_len,
        window_count,
        params_dict['sampling_rate'],
        spike_set.spike_times,
        spike_set.mean_val,
        spike_set.threshold,
    )
    fig.savefig(f'./Plots/{electrode_num:02}/bandapass_trace_snippets.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    ############################################################

    # Delete filtered electrode from memory
    del electrode

    # Dejitter these spike waveforms, and get their maximum amplitudes
    # Slices are returned sorted by amplitude polaity
    spike_set.dejitter_spikes()

    ############################################################
    # Load classifier if specificed
    classifier_params = json.load(
        open('params/waveform_classifier_params.yaml', 'r'))

    if classifier_params['use_classifier']:
        classifier_handler = bpu.classifier_handler()
        classifier_handler.load_pipelines()
        classifier_handler.classify_waveforms(
                spike_set.slices_dejittered,
                spike_set.times_dejittered,
                )

        # throw_out_noise = True
        if classifier_params['throw_out_noise']:
            # Remaining data is now only spikes
            slices_dejittered, times_dejittered, clf_prob = \
                classifier_handler.return_only_spikes()

    ############################################################

    spike_set.extract_amplitudes()
    spike_set.extract_features(
            classifier_handler.feature_pipeline,
            classifier_handler.feature_labels,
            )
    #spike_set.pca_slices()
    spike_set.create_pca_plot()
    spike_set.write_out_spike_data()

    standard_data = spike_set.feature_data

    # Set a threshold on how many datapoints are used to FIT the gmm
    # Run GMM, from 2 to max_clusters
    for i in range(max_clusters-1):
        # If dataset is very large, take subsample for fitting
        cluster_handler = bpu.cluster_handler(params_dict)
        train_set = cluster_handler.get_train_set(standard_data)
        cluster_handler.run_clustering(...)


    # Make file for dumping info about memory usage
    f= open(f'./memory_monitor_clustering/{electrode_num:02}.txt', 'w')
    print(mm.memory_usage_resource(), file=f)
    f.close()
    print(f'Electrode {electrode_num} complete.')
