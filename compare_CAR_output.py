# Import stuff!
import tables
import numpy as np
import os
import easygui
import sys
from tqdm import tqdm
import glob
import json
from utils.blech_utils import imp_metadata


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind

############################################################
############################################################

dir_name = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511/'
# Get name of directory with the data files
metadata_handler = imp_metadata([[], dir_name])
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Get CAR dirs
car_dirs = glob.glob('CAR*')

# Get all files in both CAR dirs
file_list = [sorted(os.listdir(x)) for x in car_dirs]

# Make sure files in both dirs are the same
assert file_list[0] == file_list[1]

# Make sure contents of files are the same
for file_a, file_b in zip(file_list[0], file_list[1]):
    print(f'Comparing {file_a} and {file_b}')
    dat_a = np.load(os.path.join(car_dirs[0], file_a))
    dat_b = np.load(os.path.join(car_dirs[1], file_b))
    assert np.all(dat_a == dat_b)
    print(f'Files {file_a} and {file_b} are the same!')
