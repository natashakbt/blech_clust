DIR=$1
python blech_clust.py $DIR &&
python blech_common_avg_reference.py $DIR &&
for x in $(seq 10);do bash blech_clust_jetstream_parallel.sh;done &&
bash bash_umap_spike_scatter.sh;
#rsync -avP $DIR/$(basename $DIR).h5 $DIR/$(basename $DIR).copy
