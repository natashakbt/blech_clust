# Import stuff!
import tables
import os
import numpy as np
import tqdm

# Create EArrays in hdf5 file 
#def create_hdf_arrays(file_name, ports, dig_in, emg_port, emg_channels):
def create_hdf_arrays(file_name, non_emg_channels, dig_in, emg_port, emg_channels):
        hf5 = tables.open_file(file_name, 'r+')
        #n_electrodes = len(ports)*32
        atom = tables.IntAtom()
        
        # Create arrays for digital inputs
        for i in dig_in:
                dig_inputs = hf5.create_earray(\
                        '/digital_in', 'dig_in_%i' % i, atom, (0,))

        # Create arrays for neural electrodes, and make directories to store 
        # stuff coming out from blech_process
        for i in non_emg_channels:
            el = hf5.create_earray('/raw', f'electrode{i:02}', atom, (0,))
                
        # Create arrays for EMG electrodes
        for i in range(len(emg_channels)):
            el = hf5.create_earray('/raw_emg', f'emg{i:02}', atom, (0,))

        # Close the hdf5 file 
        hf5.close()     

# Read files into hdf5 arrays - the format should be 'one file per channel'
#def read_files(hdf5_name, ports, dig_in, emg_port, emg_channels):
#        hf5 = tables.open_file(hdf5_name, 'r+')
#
#        # Read digital inputs, and append to the respective hdf5 arrays
#        print('Reading dig-ins')
#        for i in tqdm.tqdm(dig_in):
#                inputs = np.fromfile('board-DIN-%02d'%i + '.dat', 
#                                        dtype = np.dtype('uint16'))
#                exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")
#
#        # Read data from amplifier channels
#        emg_counter = 0
#        el_counter = 0
#        for port in ports:
#                print('Reading channels from Port {}'.format(port))
#                for channel in tqdm.tqdm(range(32)):
#                        data = np.fromfile('amp-' + port + '-%03d'%channel + '.dat', 
#                                dtype = np.dtype('int16'))
#                        if port == emg_port[0] and channel in emg_channels:
#                            exec(f"hf5.root.raw_emg.emg{emg_counter:02}."\
#                                    "append(data[:])")
#                            emg_counter += 1
#                        else:
#                            exec(f"hf5.root.raw.electrode{el_counter:02}."\
#                                    "append(data[:])")
#                            el_counter += 1
#                hf5.flush()
#
#        hf5.close()
        
def read_files_abu(hdf5_name, dig_in, electrode_layout_frame):
        hf5 = tables.open_file(hdf5_name, 'r+')

        # Read digital inputs, and append to the respective hdf5 arrays
        print('Reading dig-ins')
        for i in tqdm.tqdm(dig_in):
                inputs = np.fromfile('board-DIN-%02d'%i + '.dat', 
                                        dtype = np.dtype('uint16'))
                exec("hf5.root.digital_in.dig_in_"+str(i)+".append(inputs[:])")

        # Read data from amplifier channels
        emg_counter = 0
        #for port in ports:
        for num,row in tqdm.tqdm(electrode_layout_frame.iterrows()):
            print(f'Reading : {row.filename, row.CAR_group}')
            port = row.port
            # Loading should use file name 
            # but writing should use channel ind so that channels from 
            # multiple boards are written into a monotonic sequence
            channel_ind = row.electrode_ind
            data = np.fromfile(row.filename, dtype = np.dtype('int16'))
            if 'emg' in row.CAR_group.lower():
                exec(f"hf5.root.raw_emg.emg{emg_counter:02}."\
                        "append(data[:])")
                emg_counter += 1
            else:
                exec(f"hf5.root.raw.electrode{channel_ind:02}."\
                        "append(data[:])")
            hf5.flush()

        hf5.close()
