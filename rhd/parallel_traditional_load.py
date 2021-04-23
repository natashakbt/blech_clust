import numpy as np
import jax.numpy as jnp
from glob import glob
import os
from load_intan_test import *
import tables
import rhd
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange
import time

file_path = '/media/bigdata/Abuzar_Data/trad_intan_test'

file_list = sorted(glob(os.path.join(file_path,'*.rhd')))

# Get size of all files
# Or simpler yet, get size of first and last file and number of files
# All files except the last one will have the same size
first_size = get_file_info(file_list[0])
last_size = get_file_info(file_list[-1])

full_size = first_size*(len(file_list)-1) + last_size

test = rhd.read_data(file_list[0], no_floats = True)
test_data = test['amplifier_data']

atom = tables.Int16Atom()
hdf5_path = os.path.join(file_path, 'test.h5')
hf5 = tables.open_file(hdf5_path, 'w')
hf5.close()
hf5 = tables.open_file(hdf5_path, 'r+')

amp_array = hf5.create_array('/','amp_data', atom = atom, shape = (32, full_size))

#write_bool_name = 'WRITE_BOOL'
#os.environ[write_bool_name] = '0'

class write_handler():
    def __init__(self):
        pass
        #    #self.write_bool = 0
        #    os.environ['write_bool'] = '0'

    def write_off(self):
        #self.write_bool = 0
        os.environ['write_bool'] = '0'

    def write_on(self):
        #self.write_bool = 1
        os.environ['write_bool'] = '1'

    def get_write_bool(self):
        #return self.write_bool
        return int(os.environ['write_bool'])

#def write_off():
#    globals()['write_bool'] = 0
#def write_on():
#    globals()['write_bool'] = 1

def write_to_array(array, data, block_len, ind, write_handler):
    if write_handler.get_write_bool():
        write_handler.write_off()
        array[:,ind*block_len : (ind+1)*block_len] = data
        write_handler.write_on()

array_path = '/amp_data'
def write_to_file(hdf5_path, array_path, data, block_len, ind, write_handler):
    if write_handler.get_write_bool():
        write_handler.write_off()
        print("Writing now")
        print(f"{os.environ['write_bool']}")
        with tables.open_file(hdf5_path, 'r+') as hf5:
            array = hf5.get_node(\
                    os.path.dirname(array_path),os.path.basename(array_path))
            array[:,ind*block_len : (ind+1)*block_len] = data
            write_handler.write_on()
    else:
        print("Can't write currently...waiting")

#for num, name in tqdm(enumerate(file_list[:-1])):
#    data_dict = rhd.read_data(name)
#    write_to_array(array = amp_array, 
#                data = data_dict['amplifier_data'],
#                block_len = first_size,
#                ind = num,
#                write_handler = write_handler()
#                )
#
#    write_to_file(hdf5_path = hdf5_path,
#                array_path = array_path,
#                data = data_dict['amplifier_data'],
#                block_len = first_size,
#                ind = num,
#                write_handler = write_handler()
#                )

def write_to_file_par(num):
    time.sleep(0.2*num)
    write_to_file(hdf5_path = hdf5_path,
                array_path = array_path,
                data = data_dict['amplifier_data'],
                block_len = first_size,
                ind = num,
                write_handler = write_handler()
                )

#Parallel(n_jobs = 8)(delayed(write_to_array)\
#        (amp_array, data_dict['amplifier_data'], first_size, num, write_handler()) \
#        for num in range(3))

this_write_handler = write_handler()
Parallel(n_jobs = 8)(delayed(write_to_file)\
        (hdf5_path, array_path, data_dict['amplifier_data'], \
        first_size, num, this_write_handler) \
        for num in range(5))

Parallel(n_jobs = 8)(delayed(write_to_file_par)\
        (num) for num in range(2))

## ========================================##

import multiprocessing
import sys

def worker_with(lock, stream, num):
    print(f'Before lock {num}')
    with lock:
        stream.write(f'Lock acquired via with ({num})\n')
        time.sleep(10)
        
lock = multiprocessing.Lock()
w = multiprocessing.Process(target=worker_with, args=(lock, sys.stdout,0))
w1 = multiprocessing.Process(target=worker_with, args=(lock, sys.stdout,1))

w.start()
w1.start()

w.join()
w1.join()


import multiprocessing as mp
lock = mp.Lock()
pool = mp.Pool(processes=4)
results = pool.map(write_to_file_par, range(1,7))


array_path = '/amp_data'
def write_to_file(hdf5_path, array_path, data, block_len, ind, lock):
    with lock:
        print(f"Writing {ind} now")
        with tables.open_file(hdf5_path, 'r+') as hf5:
            array = hf5.get_node(\
                    os.path.dirname(array_path),os.path.basename(array_path))
            array[:,ind*block_len : (ind+1)*block_len] = data

def load_and_write(num,
                    file_list,
                    hdf5_path,
                    array_path,
                    block_len,
                    lock):
    loaded_dict = rhd.read_data(file_list[num], no_floats = True)
    loaded_dat = loaded_dict['amplifier_data']
    referenced_dat = loaded_dat - jnp.mean(loaded_dat,axis=0)
    write_to_file(hdf5_path, array_path, loaded_dat, block_len, num, lock)

def load_and_write_par(num,lock):
    load_and_write(num,
                        file_list,
                        hdf5_path,
                        array_path,
                        block_len,
                        lock)


#def write_to_file_par(num, lock):
#    print(f'Iteration {num}')
#    write_to_file(hdf5_path = hdf5_path,
#                array_path = array_path,
#                data = data_dict['amplifier_data'],
#                block_len = first_size,
#                ind = num,
#                lock = lock 
#                )

os.environ['JAX_PLATFORM_NAME'] = 'cpu'

start = time.time()
lock = mp.Lock()
process_list = [mp.Process(target = load_and_write_par, args=(num, lock)) \
                    for num in range(len(file_list)-1)]
for this_process in process_list:
    this_process.start()
for this_process in process_list:
    this_process.join()
end = time.time()
print(end-start)
