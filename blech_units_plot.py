# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
import matplotlib.pyplot as plt
import blech_waveforms_datashader
import shutil
import memory_monitor as mm

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
	shutil.rmtree("unit_waveforms_plots", ignore_errors = True)
except:
	pass
os.mkdir("unit_waveforms_plots")

# Now plot the waveforms from the units in this directory one by one
for unit in range(len(units)):
	waveforms = units[unit].waveforms[:]
	x = np.arange(waveforms.shape[1]/10) + 1
	fig, ax = blech_waveforms_datashader.waveforms_datashader(waveforms, x)
	ax.set_xlabel('Sample (30 samples per ms)')
	ax.set_ylabel('Voltage (microvolts)')
	ax.set_title('Unit %i, total waveforms = %i' % (unit, waveforms.shape[0]) + '\n' + 'Electrode: %i, Single Unit: %i, RSU: %i, FS: %i' % (hf5.root.unit_descriptor[unit]['electrode_number'], hf5.root.unit_descriptor[unit]['single_unit'], hf5.root.unit_descriptor[unit]['regular_spiking'], hf5.root.unit_descriptor[unit]['fast_spiking']))
	fig.savefig('./unit_waveforms_plots/Unit%i.png' % (unit))
	plt.close("all")
	print(mm.memory_usage_resource())

# Finally delete the folder with the temporary image files created during plotting
shutil.rmtree("datashader_temp", ignore_errors = True)

hf5.close()

