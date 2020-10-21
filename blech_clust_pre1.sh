DIR=$1
python blech_clust.py $DIR &&
python blech_common_avg_reference.py $DIR &&
bash blech_clust_jetstream_parallel.sh
