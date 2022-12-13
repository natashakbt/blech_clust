DIR=$1
BLECH_DIR=$HOME/Desktop/blech_clust
python $BLECH_DIR/blech_clust.py $DIR
python $BLECH_DIR/emg/patch/adjust_wave_threshold.py $DIR
bash $BLECH_DIR/blech_clust_jetstream_parallel.sh
python $BLECH_DIR/emg/patch/select_some_waveforms.py $DIR
python $BLECH_DIR/blech_post_process.py -d $DIR -f $DIR/*sorted_units.csv
bash $BLECH_DIR/blech_clust_post.sh $DIR
