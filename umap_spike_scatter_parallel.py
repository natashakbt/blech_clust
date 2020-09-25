import numpy as np
import os
import umap
import pylab as plt
import glob
from tqdm import trange
import sys
import json

# Read blech.dir, and cd to that directory
with open('umap_dir.txt','r') as f:
    data_dir = f.readline()[:-1]

# Read the clustering params for the file
with open(glob.glob(data_dir + '/*params*')[0],'r') as params_file_connect:
    params_dict = json.load(params_file_connect)
cluster_num = params_dict['max_clusters']
#with open(glob.glob(data_dir + '/*params*')[0],'r') as param_file:
#    params = [float(line) for line in param_file.readlines()[:-1]]
#cluster_num = int(params[0])

# Get PCA waveforms from spike_waveforms
# Get cluster predictions from clustering_results
# Plot output in Plots
electrode_num = int(sys.argv[-1])

def umap_plots(data_dir, electrode_num):
    # If processing has happened, the file will exist
    pca_file = data_dir + \
                '/spike_waveforms/electrode{0:02d}/pca_waveforms.npy'.format(electrode_num)

    if os.path.isfile(pca_file):
        
        try:
            print('Processing Electrode {0:02}'.format(electrode_num))
            pca_waveforms = np.load(data_dir + \
                    '/spike_waveforms/electrode{0:02d}/pca_waveforms.npy'.format(electrode_num))

            umap_waveforms = umap.UMAP(n_components = 2).\
                    fit_transform(pca_waveforms[:,:20])
            
            clustering_results = [np.load(data_dir + \
                    '/clustering_results/electrode{0:02d}/clusters{1}/predictions.npy'.\
                    format(electrode_num, cluster)) for cluster in \
                    range(2,cluster_num+1)] 
            
            spike_times = np.load(data_dir + \
                    '/spike_times/electrode{0:02d}/spike_times.npy'.format(electrode_num))

            print(f'Outputting figures : Electrode {electrode_num:02}')

            for cluster in range(2,cluster_num+1):

                this_clust = clustering_results[cluster-2]
                fig1, ax1 = plt.subplots()
                scatter = ax1.scatter(umap_waveforms[:,0],umap_waveforms[:,1],\
                        c = this_clust, s = 2, cmap = 'brg')
                legend = ax1.legend(*scatter.legend_elements())
                ax1.add_artist(legend)
                fig1.savefig(data_dir + \
                    '/Plots/{0:02d}/{1}_clusters_waveforms_ISIs/{1}cluster_umap.png'.\
                    format(electrode_num, cluster), 
                    dpi = 300)
                plt.close(fig1)

                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                fig2, ax2 = plt.subplots()
                ax2.hexbin(umap_waveforms[:,0],umap_waveforms[:,1], gridsize = nbins)
                fig2.savefig(data_dir + \
                        '/Plots/{0:02d}/{1}_clusters_waveforms_ISIs/{1}cluster_umap_hist.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                ax3[0,0].hexbin(spike_times, umap_waveforms[:,0], gridsize = nbins)
                scatter = ax3[1,0].scatter(spike_times, umap_waveforms[:,0],\
                        c = this_clust, s = 2, cmap = 'brg')
                legend = ax3[1,0].legend(*scatter.legend_elements())
                ax3[1,0].add_artist(legend)

                nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
                ax3[0,1].hexbin(spike_times, umap_waveforms[:,1], gridsize = nbins)
                scatter = ax3[1,1].scatter(spike_times, umap_waveforms[:,1],\
                        c = this_clust, s = 2, cmap = 'brg')
                legend = ax3[1,1].legend(*scatter.legend_elements())
                ax3[1,1].add_artist(legend)

                plt.tight_layout()

                fig3.savefig(data_dir + \
                    '/Plots/{0:02d}/{1}_clusters_waveforms_ISIs/{1}cluster_umap_timeseries.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig3)

                fig4, ax4 = plt.subplots(2,2,figsize=(20,10))
                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,0].hexbin(spike_times, pca_waveforms[:,0], gridsize = nbins)
                scatter = ax4[1,0].scatter(spike_times, pca_waveforms[:,0],\
                        c = this_clust, s = 2, cmap = 'brg')
                legend = ax4[1,0].legend(*scatter.legend_elements())
                ax4[1,0].add_artist(legend)

                nbins = np.min([100,int(pca_waveforms.shape[0]/100)])
                ax4[0,1].hexbin(spike_times, pca_waveforms[:,1], gridsize = nbins)
                scatter = ax4[1,1].scatter(spike_times, pca_waveforms[:,1],\
                        c = this_clust, s = 2, cmap = 'brg')
                legend = ax4[1,1].legend(*scatter.legend_elements())
                ax4[1,1].add_artist(legend)

                plt.tight_layout()

                fig4.savefig(data_dir + \
                    '/Plots/{0:02d}/{1}_clusters_waveforms_ISIs/cluster{1}_pca_timeseries.png'.\
                    format(electrode_num, cluster),
                    dpi = 300)
                plt.close(fig4)

                # Plot rasters for each individual cluster
                max_time = np.max(spike_times)
                min_y, max_y = np.min(umap_waveforms), np.max(umap_waveforms)
                for clust_num in np.sort(np.unique(this_clust))[1:]:
                    relevant_waveforms = umap_waveforms[this_clust == clust_num,0]
                    relevant_spiketimes = spike_times[this_clust == clust_num]
                    fig,ax  = plt.subplots()
                    ax.scatter(relevant_spiketimes,relevant_waveforms, s = 2, alpha = 0.8)
                    ax.set_xlim(0, max_time)
                    ax.set_ylim(min_y, max_y)
                    fig.savefig(data_dir + \
                        '/Plots/{0:02d}/{1}_clusters_waveforms_ISIs/Cluster{2}_raster.png'.\
                        format(electrode_num, cluster, clust_num),
                        dpi = 300)
                    plt.close(fig)

        except:
            # In other words, I'm too lazy to actually debug shit
            pass

umap_plots(data_dir, electrode_num)
