"""
Select a specified number of waveforms from each electrode present
This is intended for the emg pipeline patch
Therefore, select about 3000 waveforms from 1 electrode
"""

import sys
import os
import glob
import numpy as np
import re
import pandas as pd

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

basename = os.path.basename(dir_name[:-1])

print(f'Processing : {dir_name}')

os.chdir(dir_name)

# Load clustering results
clustering_path = os.path.join(dir_name, 'clustering_results')
pred_files = sorted(glob.glob(os.path.join(clustering_path,'*','*','*.npy')))

max_clust_pred = pred_files[-1]
preds = np.load(max_clust_pred)
splits = max_clust_pred.split('/')
electrode_str = splits[-3]
cluster_str = splits[-2]

num_pattern = '(\d+)'
electrode_num = int(re.findall(num_pattern, electrode_str)[0])
cluster_num = int(re.findall(num_pattern, cluster_str)[0])

# Find smallest combination of clusters which passes threshold
wanted_count = 3000
cluster_vals = np.unique(preds)
cluster_counts = [sum(preds==x) for x in np.unique(preds)]
count_order = np.argsort(cluster_counts)

wanted_clusts = []
total_count = 0
for i in count_order:
    this_val = cluster_vals[i]
    if this_val != -1:
        if total_count < wanted_count:
            this_count = cluster_counts[i]
            total_count += this_count
            wanted_clusts.append(this_val)

# Use post-process sheet template to write out a new sheet for this dataset
home_dir = os.getenv('HOME')
blech_clust_path = os.path.join(home_dir, 'Desktop','blech_clust')
csv_path = os.path.join(
        blech_clust_path, 
        'example_meta_files',
        'GC_PC_taste_odor_spont_210919_175343.csv')

sorting_table = pd.read_csv(csv_path, keep_default_na = False)
sorting_table = sorting_table.iloc[0]
sorting_table['Chan'] = electrode_num
sorting_table['Solution'] = cluster_num
sorting_table['Cluster'] = '+'.join([str(x) for x in wanted_clusts])
sorting_table['single_unit'] = ''
sorting_table['Type'] = ''

sorting_table = pd.DataFrame(sorting_table).T

# Write out to data folder
sorting_table.to_csv(
        os.path.join(dir_name,basename+'_sorted_units.csv'),
        index = False
        )
