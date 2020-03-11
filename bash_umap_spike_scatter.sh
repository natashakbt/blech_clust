cat blech.dir > umap_dir.txt
parallel -k -j 16 --load 100% --progress --memfree 4G --retry-failed python umap_spike_scatter_parallel.py ::: {0..63}
