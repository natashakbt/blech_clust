DIR=$1
BLECH_DIR=$HOME/Desktop/blech_clust
python $BLECH_DIR/blech_units_similarity.py $DIR;
python $BLECH_DIR/blech_units_plot.py $DIR;
python $BLECH_DIR/blech_units_make_arrays.py $DIR;
python $BLECH_DIR/blech_make_psth.py $DIR;
python $BLECH_DIR/blech_palatability_identity_setup.py $DIR;
python $BLECH_DIR/blech_overlay_psth.py $DIR;
