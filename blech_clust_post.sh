DIR=$1
python blech_units_similarity.py $DIR;
python blech_units_plot.py $DIR;
python blech_units_make_arrays.py $DIR;
python blech_make_psth.py $DIR;
python blech_palatability_identity_setup.py $DIR;
python blech_overlay_psth.py $DIR;
