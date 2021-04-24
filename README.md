# blech_clust

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis.  Optimized for the
High performance computing cluster at Brandeis
(https://kb.brandeis.edu/display/SCI/High+Performance+Computing+Cluster) but
can be easily modified to work in any parallel environment. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

# Convenience scripts
blech_clust_pre.sh : Runs steps 1-4
blech_clust_post.sh : Runs steps 6-12 

# Order of operations
0 - python blech_exp_info.py
    - Pre-clustering step. Annotate recorded channels and save experimental parameters
================================================================================
1 - python blech_clust.py 
    - Setup directories and define clustering parameters
2 - python blech_common_avg_reference.py
    - Perform common average referencing to remove large artifacts
3 - bash blech_clust_jetstream_parallel.sh
    - Embarrasingly parallel spike extraction and clustering
4 - bash bash_umap_spike_scatter.sh;
    - UMAP embedding of spikes for visualization of clusters, and generate spike-time rasters
================================================================================
5 - python blech_post_process.sh
    - Add selected units to HDF5 file for further processing
================================================================================
6 - python blech_units_similarity.py
    - Check for collisions of spiketimes to assess double-counting of waveforms in clustering
7 - python blech_units_plot.py
    - Plot waveforms of selected spikes
8 - python blech_units_make_arrays.py
    - Generate spike-train arrays
9 - python blech_make_psth.py
    - Plots PSTHs and rasters for all selected units
10- python blech_palatability_identity_setup.py
11- python blech_palatability_identity_plot.py
    - Perform ancillary analyses on spike trains by calculating:
        - Repsonsive and discriminatory neurons (ANOVA and LDA)
        - Palatability correlation coefficient
12- python blech_overlay_psth.py
    - Plot overlayed PSTHs for units with respective waveforms
