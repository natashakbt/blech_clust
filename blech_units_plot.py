# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt

# Ask for the directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Get the names of all files in the current directory, and find the hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open up the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Get all the units from the hdf5 file
units = hf5.list_nodes('/sorted_units')

# Delete and remake a directory for storing the plots of the unit waveforms (if it exists)
try:
	os.system("rm -r ./unit_waveforms_plots")
except:
	pass
os.mkdir("unit_waveforms_plots")

# Now plot the waveforms from the units in this directory one by one
for unit in range(len(units)):
	waveforms = units[unit].waveforms[:]
	fig = plt.figure()
	x = np.arange(waveforms.shape[1]/10)
	plt.plot(x - 15, waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
	plt.xlabel('Time')
	plt.ylabel('Voltage (microvolts)')
	plt.title('Unit %i, total waveforms = %i' % (unit, waveforms.shape[0]) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['electrode_number'], hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
	fig.savefig('./unit_waveforms_plots/Unit%i.png' % (unit))
	plt.close("all")

hf5.close()

