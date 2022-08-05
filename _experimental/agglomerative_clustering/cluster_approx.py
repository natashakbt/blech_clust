"""
Testing approximations to speed up clustering
"""
import matplotlib
import os
import tables
import numpy as np
import sys
import json
import pylab as plt
import matplotlib.cm as cm
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA as pca
from sklearn.mixture import GaussianMixture as gmm
from glob import glob
from clustering import *
import sys
import umap
import pandas as pd
import math
from sklearn.cluster import AgglomerativeClustering
from time import time
from scipy.cluster.hierarchy import cut_tree
from tqdm import tqdm

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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    linkage_martix = calc_linkage(model)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


data_dir = '/media/bigdata/projects/neuRecommend/test_data/raw'
data_files = sorted(glob(os.path.join(data_dir, '*.npy')))

ind = 15
#for ind in range(len(data_files)):
#slices_dejittered = np.load(data_files[ind])[:,::10]
slices_dejittered = np.load(data_files[ind])
print(slices_dejittered.shape)

plt.imshow(zscore(slices_dejittered,axis=-1), aspect='auto');plt.show()

plt.plot(slices_dejittered[::100].T);plt.show()
#plt.plot(zscore(slices_dejittered[::100].T, axis=-1));plt.show()

scaled_slices, energy = scale_waveforms(slices_dejittered)
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 1))
data[:,:n_pc] = pca_slices[:,:n_pc]
data[:,n_pc] = energy[:]/np.max(energy)

standard_data = scaler().fit_transform(data)
dat_thresh = 10e3
train_set = standard_data[np.random.choice(np.arange(standard_data.shape[0]),
                int(np.min((standard_data.shape[0],dat_thresh))))]

############################################################

num_iter = int(1e3)
num_restarts = 10
thresh = 1e-4

i = 4
# If dataset is very large, take subsample for fitting
model = gmm(
        n_components = i, 
        max_iter = num_iter, 
        n_init = num_restarts, 
        tol = thresh).fit(train_set)

predictions = model.predict(standard_data)

cluster_waveforms = [slices_dejittered[predictions==x] for x in np.unique(predictions)]

umap_waveforms = umap.UMAP(n_components = 2).\
        fit_transform(train_set)

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_waveforms.T, 
        #color = 'k', 
        c =model.predict(train_set), 
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

#plt.hist(standard_data[:,-1], bins = 30);plt.show()
#plt.imshow(standard_data, aspect='auto');plt.colorbar();plt.show()

############################################################
total_bins = 10000
bin_num = int(np.round(math.log(total_bins, standard_data.shape[1])))
print(bin_num)
standard_frame = pd.DataFrame(standard_data)
standard_frame.columns = [chr(x) for x in range(97, 97+len(standard_frame.columns))]
# Instead of even-width bins, use percentile bins
for x in standard_frame.columns:
    standard_frame[f'{x}_binned'] = pd.qcut(
                                    standard_frame[x], 
                                    q = bin_num,
                                    labels = np.arange(bin_num),
                                    retbins = True)[0]
    #standard_frame[f'{x}_binned'] = pd.cut(
    #                                standard_frame[x], 
    #                                bins = bin_num,
    #                                labels = np.arange(bin_num),
    #                                retbins = True)[0]
bin_cols = [x for x in standard_frame.columns if 'bin' in x]
#bin_frame = standard_frame[bin_cols].drop_duplicates()
standard_binned_frame = standard_frame.groupby(bin_cols).mean().dropna() 
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns = bin_cols)
standard_binned_frame.shape

binned_model = gmm(
        n_components = i, 
        max_iter = num_iter, 
        n_init = num_restarts, 
        tol = thresh).fit(standard_binned_frame.to_numpy())

binned_predictions = binned_model.predict(standard_data)
binned_cluster_waveforms = [slices_dejittered[binned_predictions==x] for x in np.unique(binned_predictions)]

binned_umap = umap.UMAP(n_components = 2).\
        fit(standard_binned_frame)
umap_binned_waveforms = binned_umap.transform(standard_data) 

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c =binned_model.predict(standard_data), 
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, binned_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Hierarchical clustering on binned data


ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

agg_bin_frame = bin_frame.copy()
agg_bin_frame['labels'] = binned_agg_predictions
agg_standard_frame = standard_frame.copy()

fin_agg_frame = agg_standard_frame.merge(agg_bin_frame)

agg_predictions = fin_agg_frame.labels
agg_cluster_waveforms = [slices_dejittered[agg_predictions==x] for x in np.unique(agg_predictions)]

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c = agg_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, agg_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Hierarchical clustering on standard data (no binning)

ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_data)
agg_predictions = ward.labels_

#agg_bin_frame = bin_frame.copy()
#agg_bin_frame['labels'] = binned_agg_predictions
#agg_standard_frame = standard_frame.copy()
#
#fin_agg_frame = agg_standard_frame.merge(agg_bin_frame)
#
#agg_predictions = fin_agg_frame.labels
agg_cluster_waveforms = [slices_dejittered[agg_predictions==x] for x in np.unique(agg_predictions)]

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(i,2,x+2) for x in np.arange(2*i, step = 2)]
scatter = ax0.scatter(*umap_binned_waveforms.T, 
        #color = 'k', 
        c = agg_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, agg_cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

############################################################
# Binned Hierarchical clustering on UMAP data

start_t = time()
# UMAP data, this should "pre-cluster" it
umap_obj = umap.UMAP(n_components = 2).fit(train_set)
umap_waveforms = umap_obj.transform(standard_data)

# Bin the UMAP data
total_bins = 10000
bin_num = int(np.round(total_bins ** (1/umap_waveforms.shape[1])))
print(bin_num)
umap_frame = pd.DataFrame(umap_waveforms)
umap_frame.columns = [chr(x) for x in range(97, 97+len(umap_frame.columns))]
# Instead of even-width bins, use percentile bins
for x in umap_frame.columns:
    umap_frame[f'{x}_binned'] = pd.qcut(
                                    umap_frame[x], 
                                    q = bin_num,
                                    labels = np.arange(bin_num),
                                    retbins = True)[0]
    #umap_frame[f'{x}_binned'] = pd.cut(
    #                                umap_frame[x], 
    #                                bins = bin_num,
    #                                labels = np.arange(bin_num),
    #                                retbins = True)[0]
bin_cols = [x for x in umap_frame.columns if 'bin' in x]
non_bin_cols = [x for x in umap_frame.columns if x not in bin_cols]
#bin_frame = umap_frame[bin_cols].drop_duplicates()
standard_binned_frame = umap_frame.groupby(bin_cols).mean().dropna() 
standard_binned_frame = standard_binned_frame.reset_index(drop=False)
bin_frame = standard_binned_frame[bin_cols]
standard_binned_frame = standard_binned_frame.drop(columns = bin_cols)
standard_binned_frame.shape

#fig,ax = plt.subplots(1,2)
#ax[0].scatter(*umap_waveforms.T, alpha = 0.7)
#ax[1].scatter(*standard_binned_frame.to_numpy().T, alpha = 0.7)
#plt.show()

#ward = AgglomerativeClustering(n_clusters=i, linkage="ward").fit(standard_binned_frame)
ward = AgglomerativeClustering(
        distance_threshold =0, 
        n_clusters = None, 
        linkage="ward").fit(standard_binned_frame)
binned_agg_predictions = ward.labels_

linkage = calc_linkage(ward)

#dendrogram(linkage,truncate_mode="level", p=3)
#dendrogram(linkage,truncate_mode="lastp", p=7)
#plt.show()

n_clusters = 7
cut_pred = cut_tree(linkage, n_clusters = n_clusters).flatten()

clust_range = np.arange(2,8)
clust_label_list = [
        cut_tree(
            linkage, 
            n_clusters = this_num
            ).flatten()
        for this_num in tqdm(clust_range)
        ]


cut_label_array = np.stack(clust_label_list)

#plt.scatter(*standard_binned_frame.to_numpy().T, c = cut_pred)
#plt.show()

label_cols = [f'label_{i}' for i in clust_range]
bin_frame[label_cols] = pd.DataFrame(cut_label_array.T)#ward.labels_
fin_agg_frame = umap_frame.merge(bin_frame, how = 'left')

fin_label_cols = fin_agg_frame[label_cols]



def register_labels(x,y):
    #x = fin_label_cols.iloc[:,2]
    #y = fin_label_cols.iloc[:,3]
    unique_x = np.unique(x)
    x_cluster_y = [np.unique(y[x==i]) for i in unique_x]
    return dict(zip(unique_x, x_cluster_y))

map_dict = {}
for i in range(len(clust_range)-1):
    map_dict[i] = register_labels(
            fin_label_cols.iloc[:,i],
            fin_label_cols.iloc[:,i+1]
            )

from matplotlib.patches import ConnectionPatch

cmap = plt.get_cmap('tab10')
slice_mid = slices_dejittered.shape[1]//2
fig,ax = plt.subplots(len(clust_range), np.max(clust_range))
for row in range(len(clust_range)):
    #row = 5
    center = int(np.max(clust_range)//2)
    labels = fin_label_cols.T.iloc[row]
    unique_labels = np.unique(labels)
    med_label = int(np.median(unique_labels))
    ax_inds = unique_labels - med_label + center
    parent_unique_labels = np.unique(fin_label_cols.T.iloc[row-1])
    parent_med_label = int(np.median(parent_unique_labels))
    parent_ax_inds = parent_unique_labels - parent_med_label + center
    ax[row,0].set_ylabel(f'{clust_range[row]} Clusters')
    for x in unique_labels:
        this_dat = slices_dejittered[labels==x] 
        ax[row, ax_inds[x]].plot(this_dat[np.random.choice(len(this_dat), 200)].T,
            color = cmap(0), alpha = 0.15)
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
#plt.show()

#fin_label_cols.colnames = [str(x) for x in clust_range

#fig,ax = plt.subplots(1,3)
#ax[0].scatter(*umap_waveforms.T, alpha = 0.7)
#ax[1].scatter(*standard_binned_frame.to_numpy().T, alpha = 0.7, c = binned_agg_predictions)
#ax[2].scatter(*fin_agg_frame[non_bin_cols].to_numpy().T, alpha = 0.7, c = fin_agg_frame['labels'])
#plt.show()

fin_predictions = fin_label_cols.iloc[:,-1]#fin_agg_frame['labels']
cluster_waveforms = [slices_dejittered[fin_predictions==x] for x in np.unique(fin_predictions)]

end_t = time()
time_taken = end_t - start_t
print(time_taken)

fig = plt.figure()
ax0 = plt.subplot(1, 2, 1)
ax_list = [plt.subplot(n_clusters,2,x+2) for x in np.arange(2*n_clusters, step = 2)]
scatter = ax0.scatter(*umap_waveforms.T, 
        #color = 'k', 
        c = fin_predictions,
        cmap = 'brg',
        )
        #alpha = 0.1)
legend1 = ax0.legend(*scatter.legend_elements())
ax0.add_artist(legend1)
for this_ax, this_clust in zip(ax_list, cluster_waveforms):
    this_ax.plot(this_clust[np.random.choice(len(this_clust), 30)].T,
            color = 'k', alpha = 0.5)
plt.show()

#from scipy.cluster.hierarchy import dendrogram
#plot_dendrogram(ward, truncate_mode = 'level', p = 4)


#plt.show()
