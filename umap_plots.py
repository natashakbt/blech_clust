import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import numpy as np
import pylab as plt
from tqdm import tqdm
import sys

import pandas as pd
import re
from glob import glob
from utils.blech_utils import imp_metadata
import subprocess
from itertools import combinations
from joblib import load, delayed, Parallel, cpu_count


############################################################
# Load UMAP Model
############################################################
def return_transformers():
    home_dir = os.environ.get("HOME")
    model_dir = f'{home_dir}/Desktop/neuRecommend/model'
    sys.path.append(f'{home_dir}/Desktop/neuRecommend/src/create_pipeline')
    import feature_engineering_pipeline as fep
    zscore_custom = fep.zscore_custom

    # Run download model script to make sure latest model is being used
    process = subprocess.Popen(
        f'python {home_dir}/Desktop/blech_clust/utils/download_wav_classifier.py', shell=True)
    # Forces process to complete before proceeding
    stdout, stderr = process.communicate()
    # If model_dir still doesn't exist, then throw an error
    if not os.path.exists(model_dir):
        raise Exception("Couldn't download model")

    feature_pipeline_path = f'{model_dir}/feature_engineering_pipeline.dump'
    umap_model_path = f'{model_dir}/umap_model'

    umap_model = load_ParametricUMAP(umap_model_path)
    feature_transformer = load(feature_pipeline_path)
    return feature_transformer, umap_model, zscore_custom


def return_raw_data(this_row):
    spike_waveforms = np.load(this_row.waveform_files)
    spike_times = np.load(this_row.spiketime_files)
    clustering_results = np.load(this_row.cluster_files)
    return spike_waveforms, spike_times, clustering_results


def process_umap(umap_model, spike_waveforms, feature_transformer):
    X = feature_transformer.transform(spike_waveforms)
    umap_waveforms = umap_model.transform(X)
    return umap_waveforms


def gen_cluster_output_dir(this_row, dir_name):
    cluster_output_dir = os.path.join(
        dir_name, 'Plots', f'{this_row.electrode_nums:02d}',
        f'{this_row.cluster_nums}_clusters_waveforms_ISIs')
    return cluster_output_dir


def create_cluster_plot(umap_waveforms, clustering_results, cluster_output_dir):
    """
    Creates a plot of the UMAP embedding of the spike waveforms colored by
    cluster assignment. This plot is saved to the cluster_output_dir.
    """
    umap_dims = umap_waveforms.shape[1]
    dim_combos = list(combinations(range(umap_dims), 2))
    fig, ax = plt.subplots(umap_dims, 2, sharex=True, sharey=True,
                           figsize=(8, 10))
    for i in range(umap_dims):
        scatter = ax[i, 0].scatter(umap_waveforms[:, dim_combos[i][0]],
                                   umap_waveforms[:, dim_combos[i][1]],
                                   c=clustering_results, s=5, cmap='tab10',
                                   alpha=0.5)
        legend = ax[i, 0].legend(*scatter.legend_elements(),
                                 bbox_to_anchor=(0, 1.1))
        ax[i, 0].add_artist(legend)
        ax[i, 0].set_xlabel('UMAP Dim {}'.format(dim_combos[i][0]))
        ax[i, 0].set_ylabel('UMAP Dim {}'.format(dim_combos[i][1]))
        ax[i, 1].hexbin(umap_waveforms[:, dim_combos[i][0]],
                        umap_waveforms[:, dim_combos[i][1]],
                        bins='log', cmap='Greys')
        ax[i, 1].set_xlabel('UMAP Dim {}'.format(dim_combos[i][0]))
        ax[i, 1].set_ylabel('UMAP Dim {}'.format(dim_combos[i][1]))
    fig.suptitle('UMAP Clusters')
    # plt.show()
    fig.savefig(os.path.join(cluster_output_dir, 'UMAP_clusters.png'))
    plt.close(fig)


def create_umap_time_plot(umap_waveforms, spike_times,
                          clustering_results, cluster_output_dir):
    """
    Creates a plot of UMAP embedding timeseries colored by cluster assignment.
    This plot is saved to the cluster_output_dir.
    Put legend outside plot
    """
    umap_dims = umap_waveforms.shape[1]
    fig, ax = plt.subplots(umap_dims, 2, sharex=True, sharey=True,
                           figsize=(8, 10))
    for i in range(umap_dims):
        scatter = ax[i, 0].scatter(
            spike_times, umap_waveforms[:, i],
            c=clustering_results, s=5, cmap='tab10',
            alpha=0.5,)
        legend = ax[i, 0].legend(*scatter.legend_elements(),
                                 bbox_to_anchor=(0, 1.1))
        ax[i, 0].add_artist(legend)
        ax[i, 0].set_xlabel('Time (s)')
        ax[i, 0].set_ylabel('UMAP Dim {}'.format(i))
        ax[i, 1].hexbin(spike_times, umap_waveforms[:, i],
                        bins='log', cmap='Greys')
        ax[i, 1].set_xlabel('Time (s)')
        ax[i, 1].set_ylabel('UMAP Dim {}'.format(i))
    fig.suptitle('UMAP Clusters Timeseries')
    # plt.show()
    fig.savefig(os.path.join(cluster_output_dir,
                'UMAP_cluster_timeseries.png'))
    plt.close(fig)


def run_pipeline(this_row, dir_name):
    print(f'Processing : Electrode {this_row.electrode_nums:02d}, '
          f'{this_row.cluster_nums} clusters')
    feature_transformer, umap_model, zscore_custom = return_transformers()
    spike_waveforms, spike_times, clustering_results = return_raw_data(
        this_row)
    umap_waveforms = process_umap(
        umap_model, spike_waveforms, feature_transformer)
    cluster_output_dir = gen_cluster_output_dir(this_row, dir_name)
    create_cluster_plot(umap_waveforms, clustering_results, cluster_output_dir)
    create_umap_time_plot(umap_waveforms, spike_times,
                          clustering_results, cluster_output_dir)


def return_path_frame(dir_name):
    # Iterate over all electrodes in data_dir
    waveform_files = glob(os.path.join(
        dir_name, 'spike_waveforms', "**", "spike_waveforms.npy"))
    spiketime_files = glob(os.path.join(
        dir_name, 'spike_times', "**", "spike_times.npy"))
    cluster_files = glob(os.path.join(
        dir_name, 'clustering_results', "**", "**", "predictions.npy"))
    waveform_files = sorted(waveform_files)
    spiketime_files = sorted(spiketime_files)
    cluster_files = sorted(cluster_files)
    cluster_electrodes = [re.findall('electrode\d+', x)[0]
                          for x in cluster_files]
    cluster_electrode_int = [int(x.split('electrode')[1])
                             for x in cluster_electrodes]
    cluster_nums = [re.findall('clusters\d+', x)[0] for x in cluster_files]
    cluster_num_ints = [int(x.split('clusters')[1]) for x in cluster_nums]
    waveform_files = sorted(waveform_files*len(np.unique(cluster_nums)))
    spiketime_files = sorted(spiketime_files*len(np.unique(cluster_nums)))

    path_frame = pd.DataFrame(
        dict(
            waveform_files=waveform_files,
            spiketime_files=spiketime_files,
            cluster_files=cluster_files,
            cluster_electrodes=cluster_electrodes,
            cluster_nums=cluster_num_ints,
            electrode_nums=cluster_electrode_int
        ),
        index=cluster_electrode_int
    )
    return path_frame


if __name__ == '__main__':
    ############################################################
    # Load Data
    ############################################################
    home_dir = os.environ.get("HOME")
    blech_path = f'{home_dir}/Desktop/blech_clust'
    dir_name = open(os.path.join(blech_path, 'blech.dir'), 'r').read()
    dir_name = dir_name.strip()

    #metadata_handler = imp_metadata(sys.argv)
    metadata_handler = imp_metadata([[], dir_name])
    dir_name = metadata_handler.dir_name
    print(f'Processing : {dir_name}')
    # os.chdir(dir_name)

    info_dict = metadata_handler.info_dict
    params_dict = metadata_handler.params_dict
    cluster_num = params_dict['max_clusters']

    path_frame = return_path_frame(dir_name)

    bash_file = 'bash_umap_parallel.sh'
    runner_path = os.path.join(blech_path, bash_file)
    if not os.path.exists(runner_path):
        print(f'{bash_file} does not exist')
        print(f'Creating {bash_file}')
        job_count = cpu_count()-2
        f = open(runner_path, 'w')
        print(f"parallel -k -j {job_count} --noswap --load 100% --progress " +
              "--memfree 4G --retry-failed " +
              f"--joblog {dir_name}/umap_results.log " +
              f"python umap_plots.py " +
              f"::: {' '.join([str(x) for x in range(len(path_frame))])}",
              file=f)
        f.close()
        print(f'Run {bash_file} to generate UMAP plots')
        sys.exit()

    model_dir = f'{home_dir}/Desktop/neuRecommend/model'
    sys.path.append(f'{home_dir}/Desktop/neuRecommend/src/create_pipeline')
    from umap.parametric_umap import load_ParametricUMAP
    from feature_engineering_pipeline import *
    # Run parallel pipeline manually as it needs imports from
    # feature_engineering_pipeline in __main__
    this_row = path_frame.iloc[int(sys.argv[1])]
    run_pipeline(this_row, dir_name)
    # Parallel(n_jobs=2)(delayed(run_pipeline)(this_row[1], dir_name) \
    #        for this_row in tqdm(path_frame.iterrows()))
