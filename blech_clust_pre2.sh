DIR=$1
bash bash_umap_spike_scatter.sh;
rsync -avP $DIR/$(basename $DIR).h5 $DIR/$(basename $DIR).copy
