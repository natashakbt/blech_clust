import matplotlib
import os
import tables
import numpy as np
import sys
import pylab as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler
from glob import glob
sys.path.append('/home/abuzarmahmood/Desktop/blech_clust')
import sys
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram
from tqdm import tqdm
from matplotlib.patches import ConnectionPatch

def register_labels(x,y):
    unique_x = np.unique(x)
    x_cluster_y = [np.unique(y[x==i]) for i in unique_x]
    return dict(zip(unique_x, x_cluster_y))

def calc_linkage(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix

# Resort mapping so that children are always in order
def sort_label_array(label_array):
    """
    Resort map_dict so that children are always in order
    We will need a remapping at every level

    Input:
        label_array: levels x samples

    Output:
        label_array: levels x samples
    """
    n_levels = label_array.shape[0]
    level_pairs = list(zip(np.arange(n_levels-1), np.arange(1,n_levels)))
    for x,y in level_pairs:
        parent_labels = np.unique(label_array[x])
        child_labels = np.unique(label_array[y])
        child_map = {}
        highest_so_far = 0
        for this_parent in parent_labels:
            this_child = label_array[y][label_array[x] == this_parent]
            this_child = np.unique(this_child)
            wanted_labels = np.arange(highest_so_far, highest_so_far + len(this_child))
            highest_so_far = np.max(wanted_labels) + 1 
            for i in range(len(this_child)):
                child_map[this_child[i]] = wanted_labels[i]
        # Remap
        for i in range(len(label_array[y])):
            label_array[y][i] = child_map[label_array[y][i]]
    return label_array

def perform_agg_clustering(features, max_clusters = 8):
    clust_range = np.arange(1, max_clusters+1)
    ward = AgglomerativeClustering(
            distance_threshold =0, 
            n_clusters = None, 
            linkage="ward").fit(features)
    linkage = calc_linkage(ward)
    clust_label_list = [
            cut_tree(
                linkage, 
                n_clusters = this_num
                ).flatten()
            for this_num in tqdm(clust_range)
            ]

    cut_label_array = np.stack(clust_label_list)
    cut_label_array = sort_label_array(cut_label_array)

    map_dict = {}
    for i in range(len(clust_range)-1):
        map_dict[i] = register_labels(
                cut_label_array[i],
                cut_label_array[i+1]
                )
    return cut_label_array, map_dict, clust_range

def plot_waveform_dendogram(data, cut_label_array, clust_range, plot_n = 1000,
                            save_path = None):
    if save_path is None:
        raise ValueError('Please provide a save path')
    slice_mid = data.shape[1]//2
    fig,ax = plt.subplots(len(clust_range), np.max(clust_range),
                          figsize = (7,7), sharex = True, sharey = True)
    for row in range(len(clust_range)):
        center = int(np.max(clust_range)//2)
        labels = cut_label_array.T[:,row]
        unique_labels = np.unique(labels)
        med_label = int(np.median(unique_labels))
        ax_inds = unique_labels - med_label + center
        parent_unique_labels = np.unique(cut_label_array.T[:,row-1])
        parent_med_label = int(np.median(parent_unique_labels))
        parent_ax_inds = parent_unique_labels - parent_med_label + center
        ax[row,0].set_ylabel(f'{clust_range[row]} Clusters')
        for x in unique_labels:
            this_dat = data[labels==x] 
            if len(this_dat) > plot_n:
                plot_dat = this_dat[np.random.choice(len(this_dat), plot_n)]
            ax[row, ax_inds[x]].plot(plot_dat.T,
                color = 'k', alpha = 0.01)
        if row > 0: 
            this_map = map_dict[row-1]
            for key,val in this_map.items():
                for child in val:
                    con = ConnectionPatch(
                            xyA = (slice_mid,0), coordsA = ax[row-1, parent_ax_inds[key]].transData,
                            xyB = (slice_mid,0), coordsB = ax[row, ax_inds[child]].transData,
                            arrowstyle = "-|>"
                            )
                    fig.add_artist(con)
    # Remove box round each subplot
    for ax0 in ax.flatten():
        ax0.axis('off')
    fig.savefig(save_path, dpi = 300)
    plt.close(fig)

def trim_data(data, n_max = 20000):
    """
    Agglomerative clustering doesn't like large datasets
    If we have more than n_max samples, we will randomly sample n_max samples
    
    Input:
        data: samples x features
        n_max: maximum number of samples to use

    Output:
        data: samples x features
    """
    if len(data) > n_max:
        data = data[np.random.choice(len(data), n_max, replace = False)]
    return data

############################################################
# Load data
data_dir = '/media/bigdata/projects/neuRecommend/data/sorted/pos'
data_files = sorted(glob(os.path.join(data_dir, '*.npy')))

ind = np.arange(10)
data = np.concatenate([np.load(data_files[x]) for x in ind])
data = trim_data(data)

# Get features
pca_obj = pca(n_components=2)
pca_obj.fit(data)
raw_features = pca_obj.transform(data)

scaler = StandardScaler()
scaler.fit(raw_features)
features = scaler.transform(raw_features)

plt.imshow(features.T, aspect = 'auto')
plt.show()

############################################################
# Cluster
save_path = os.path.join('/home/abuzarmahmood/Desktop', 'agg_clust.png')
cut_label_array, map_dict, clust_range = perform_agg_clustering(features, max_clusters = 7)
plot_waveform_dendogram(data, cut_label_array, clust_range, plot_n = 1000)
