"""
For data that has already undergone spike-sorting, this script removes the raw data and sorted units from the HDF5 file
"""

import tables
import numpy
from tqdm import tqdm

# First, get the size of the these datasets
path_files = '/media/bigdata/Abuzar_Data/bla_gc_and_bla_h5_paths.txt'
file_list = [x.strip() for x in open(path_files, 'r').readlines()]

contains_raw = []
for file in file_list:
    file = file_list[2]
    print(file)
    h5file = tables.open_file(file, mode='r')
    contains_raw.append(h5file.__contains__('/raw'))
    h5file.close()

print(f'{sum(contains_raw)}/{len(contains_raw)} files contain raw data')

# Remove raw data
for file in file_list:
    print(file)
    h5file = tables.open_file(file, mode='a')
    if h5file.__contains__('/raw'):
        h5file.remove_node('/', 'raw', recursive=True)
    h5file.close()

# Remove sorted units
for file in tqdm(file_list):
    print(file)
    h5file = tables.open_file(file, mode='a')
    if h5file.__contains__('/sorted_units'):
        h5file.remove_node('/', 'sorted_units', recursive=True)
    h5file.close()
