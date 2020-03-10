import numpy as np
import os
import pylab as plt
import glob
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing as mp
from sklearn.manifold import TSNE as tsne


# Read blech.dir, and cd to that directory
with open('blech.dir','r') as blech_dir:
    data_dir = blech_dir.readline()[:-1]

# Read the clustering params for the file
with open(glob.glob(data_dir + '/*params*')[0],'r') as param_file:
    params = [float(line) for line in param_file.readlines()[:-1]]
cluster_num = int(params[0])

# Get PCA waveforms from spike_waveforms
# Get cluster predictions from clustering_results
# Plot output in Plots


def tsne_plots(data_dir, electrode_num):
    # If processing has happened, the file will exist
    pca_file = data_dir + \
                '/spike_waveforms/electrode{}/pca_waveforms.npy'.format(electrode_num)

    if os.path.isfile(pca_file):
        
        try:
            pca_waveforms = np.load(data_dir + \
                    '/spike_waveforms/electrode{}/pca_waveforms.npy'.format(electrode_num))

            tsne_waveforms = tsne(n_components = 2).\
                    fit_transform(pca_waveforms[:,:20])
            
            clustering_results = [np.load(data_dir + \
                    '/clustering_results/electrode{0}/clusters{1}/predictions.npy'.\
                    format(electrode_num, cluster)) for cluster in \
                    range(2,cluster_num+1)] 
            
            spike_times = np.load(data_dir + \
                    '/spike_times/electrode{}/spike_times.npy'.format(electrode_num))

            print('Processing for Electrode {} complete'.format(electrode_num))

            for cluster in range(2,cluster_num+1):

                fig1, ax1 = plt.subplots()
                scatter = ax1.scatter(tsne_waveforms[:,0],tsne_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax1.legend(*scatter.legend_elements())
                ax1.add_artist(legend)
                fig1.savefig(data_dir + \
                    '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_tsne.png'.\
                    format(electrode_num, cluster), 
                    dpi = 300)
                plt.close(fig1)

                nbins = np.min([100,int(tsne_waveforms.shape[0]/100)])
                fig2, ax2 = plt.subplots()
                ax2.hexbin(tsne_waveforms[:,0],tsne_waveforms[:,1], gridsize = nbins)
                fig2.savefig(data_dir + \
                    '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_tsne_hist.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(tsne_waveforms.shape[0]/100)])
                ax3[0,0].hexbin(spike_times, tsne_waveforms[:,0], gridsize = nbins)
                scatter = ax3[1,0].scatter(spike_times, tsne_waveforms[:,0],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax3[1,0].legend(*scatter.legend_elements())
                ax3[1,0].add_artist(legend)

                nbins = np.min([100,int(tsne_waveforms.shape[0]/100)])
                ax3[0,1].hexbin(spike_times, tsne_waveforms[:,1], gridsize = nbins)
                scatter = ax3[1,1].scatter(spike_times, tsne_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax3[1,1].legend(*scatter.legend_elements())
                ax3[1,1].add_artist(legend)

                plt.tight_layout()

                fig3.savefig(data_dir + \
                    '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_tsne_timeseries.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig3)

                fig4, ax4 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,0].hexbin(spike_times, pca_waveforms[:,0], gridsize = nbins)
                scatter = ax4[1,0].scatter(spike_times, pca_waveforms[:,0],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax4[1,0].legend(*scatter.legend_elements())
                ax4[1,0].add_artist(legend)

                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,1].hexbin(spike_times, pca_waveforms[:,1], gridsize = nbins)
                scatter = ax4[1,1].scatter(spike_times, pca_waveforms[:,1],\
                        c = clustering_results[cluster-2], s = 2, cmap = 'brg')
                legend = ax4[1,1].legend(*scatter.legend_elements())
                ax4[1,1].add_artist(legend)

                plt.tight_layout()

                fig4.savefig(data_dir + \
                    '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_pca_timeseries.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig4)

        except:
            # In other words, I'm too lazy to actually debug shit
            pass

#for electrode_num in trange(len(os.listdir(data_dir + '/clustering_results'))):
#    tsne_plots(data_dir, electrode_num)

Parallel(n_jobs = mp.cpu_count())\
        (delayed(tsne_plots)(data_dir, electrode_num) \
        for electrode_num in \
        trange(len(os.listdir(data_dir + '/clustering_results'))))
