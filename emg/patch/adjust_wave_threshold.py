"""
Adjust waveform threshold so we get more waveforms
"""
import sys
import json
from glob import glob
import os

# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

print(f'Processing : {dir_name}')

params_path = glob(os.path.join(dir_name,"*.params")) 

if len(params_path) == 0:
    raise Exception('No params file found, run blech_clust.py first')
elif len(params_path) == 1:
    params_path = params_path[0]
else:
    raise Exception("Multiple params files found...something is wrong")

params_dict = json.load(open(params_path,'r'))
params_dict["waveform_threshold"] = 3

json.dump(
        params_dict,
        open(params_path,'w'),
        indent = 4)
