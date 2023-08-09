"""
Utilities to help with quality assurance
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import tables
import os

def get_all_channels(hf5_path, downsample_rate = 100):
	"""
	Get all channels in a file from nodes ['raw','raw_emg']

	Input:
		hf5_path: str, path to hdf5 file
		downsample_rate: int, downsample rate of data

	Output:
		all_chans: np.array (n_chans, n_samples)
		chan_names: np.array (n_chans,)
	"""
	hf5 = tables.open_file(hf5_path, 'r')
	raw = hf5.list_nodes('/raw')
	raw_emg = hf5.list_nodes('/raw_emg')
	all_chans = []
	chan_names = []
	for node in [raw, raw_emg]:
		for chan in tqdm(node):
			all_chans.append(chan[:][::downsample_rate])
			chan_names.append(chan._v_name)
	hf5.close()
	chan_names = [int(x.split('electrode')[-1]) for x in chan_names]
	# Sort everything by channel number
	sort_order = np.argsort(chan_names)
	chan_names = np.array(chan_names)[sort_order]
	all_chans = np.stack(all_chans)[sort_order]
	return all_chans, np.array(chan_names)

def intra_corr(X):
	"""
	Correlations between all channels in X

	Input:
		X: np.array (n_chans, n_samples)

	Output:
		corr_mat: np.array (n_chans, n_chans)
	"""
	inds = list(combinations(range(X.shape[0]), 2))
	corr_mat = np.zeros((X.shape[0], X.shape[0]))
	for i,j in tqdm(inds):
		corr_mat[i,j] = pearsonr(X[i,:], X[j,:])[0]
		corr_mat[j,i] = np.nan 
	return corr_mat

def gen_corr_output(corr_mat, plot_dir, thresholds = [0.9,0.93,0.97]):
	"""
	Generate a plot of the raw, and thresholded correlation matrices

	Input:
		corr_mat: np.array (n_chans, n_chans)
	
	Output:
		fig: matplotlib figure
	"""
	thresh_corr = corr_mat.copy()
	thresh_corr[thresh_corr < thresholds[0]] = np.nan
	for this_thresh in thresholds:
		thresh_corr[corr_mat >= this_thresh] = this_thresh

	save_path = os.path.join(plot_dir, 'raw_channel_corr_plot.png')
	fig, ax = plt.subplots(1,2, figsize = (10,5))
	im = ax[0].imshow(corr_mat, cmap = 'jet', vmin = 0, vmax = 1)
	ax[0].set_title('Raw Correlation Matrix')
	ax[0].set_xlabel('Channel')
	ax[0].set_ylabel('Channel')
	fig.colorbar(im, ax = ax[0])
	im = ax[1].imshow(thresh_corr, cmap = 'jet')
	ax[1].set_title('Thresholded Correlation Matrix')
	ax[1].set_xlabel('Channel')
	ax[1].set_ylabel('Channel')
	# Generate discrete colorbar given thresholds
	norm = plt.Normalize(thresholds[0], thresholds[-1])
	ticks = [thresholds[0]] + thresholds[1:-1] + [thresholds[-1]]
	cmap = plt.cm.jet
	bounds = np.linspace(thresholds[0], thresholds[-1], len(ticks))
	cmaplist = [cmap(i) for i in range(cmap.N)]
	for i in range(len(bounds)-1):
		cmaplist[i] = (1,1,1,1)
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	ax[1].imshow(thresh_corr, 
			  interpolation = 'nearest', cmap = cmap, norm = norm)
	cbar = fig.colorbar(im, ax = ax[1], ticks = ticks)
	cbar.ax.set_yticklabels(['> %.2f'%x for x in ticks])
	fig.tight_layout()
	fig.savefig(save_path)
	plt.close(fig)

	# Also output a table with only the thresholded values
	upper_thresh_corr = thresh_corr.copy()
	upper_thresh_corr[np.tril_indices_from(upper_thresh_corr, k = 1)] = np.nan
	# Convert to pd.DataFrame
	inds = np.array(list(np.ndindex(upper_thresh_corr.shape)))
	upper_thresh_frame = pd.DataFrame(
			dict(
				chan1 = inds[:,0],
				chan2 = inds[:,1],
				corr = upper_thresh_corr.flatten()
				)
			)
	upper_thresh_frame = upper_thresh_frame.dropna()
	upper_thresh_frame.to_csv(
			os.path.join(plot_dir, 'raw_channel_corr_table.txt'),
						   index = False, sep = '\t')

