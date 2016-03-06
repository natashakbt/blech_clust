import os
import tables
import numpy as np
import easygui
import ast

# Get directory where the hdf5 file sits, and change to that directory
dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Clean up the memory monitor files, pass if clean up has been done already
if not os.path.exists('./memory_monitor_clustering/memory_usage.txt'):
	file_list = os.listdir('./memory_monitor_clustering')
	f = open('./memory_monitor_clustering/memory_usage.txt', 'w')
	for files in file_list:
		try:
			mem_usage = np.loadtxt('./memory_monitor_clustering/' + files)
			print>>f, 'electrode'+files[:-4], '\t', str(mem_usage)+'MB'
			os.system('rm ' + './memory_monitor_clustering/' + files)
		except:
			pass	
	f.close()

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.openFile(hdf5_name, 'r+')

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
try:
	hf5.removeNode('/raw', recursive = 1)
	# And if successful, close the currently open hdf5 file and ptrepack the file
	hf5.close()
	print "Raw recordings removed"
	os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
	# Delete the old (raw and big) hdf5 file
	os.system("rm " + hdf5_name)
	# And open the new, repacked file
	hf5 = tables.openFile(hdf5_name[:-3] + "_repacked.h5", 'r+')
	print "File repacked"
except:
	print "Raw recordings have already been removed, so moving on .."

# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
	hf5.createGroup('/', 'sorted_units')
except:
	pass

# Define a unit_descriptor class to be used to add things (anything!) about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
	electrode_number = tables.Int32Col()
	single_unit = tables.Int32Col()

# Make a table under /sorted_units describing the sorted units. If unit_descriptor already exists, just open it up in the variable table
try:
	table = hf5.createTable('/', 'unit_descriptor', description = unit_descriptor)
except:
	table = hf5.root.unit_descriptor

# Run an infinite loop as long as the user wants to pick clusters from the electrodes	
while True:
	# Get electrode number from user
	electrode_num = easygui.multenterbox(msg = 'Which electrode do you want to choose? Hit cancel to exit', fields = ['Electrode #'])
	# Break if wrong input/cancel command was given
	try:
		electrode_num = int(electrode_num[0])
	except:
		break
	
	# Get the number of clusters in the chosen solution
	num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, fields = ['Number of clusters in the solution'])
	num_clusters = int(num_clusters[0])

	# Get cluster choices from the chosen solution
	clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', choices = tuple([str(i) for i in range(num_clusters)]))
	
	# Check if the user wants to merge clusters if more than 1 cluster was chosen
	merge = False
	if len(clusters) > 1:
		merge = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
		merge = ast.literal_eval(merge[0])

	# Get list of existing nodes/groups under /sorted_units
	node_list = hf5.listNodes('/sorted_units')

	# If node_list is empty, start naming units from 001
	unit_name = ''
	max_unit = 0
	if node_list == []:		
		unit_name = 'unit%03d' % 1
	# Else name the new unit by incrementing the last unit by 1 
	else:
		unit_numbers = []
		for node in node_list:
			unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
			unit_numbers[-1] = int(unit_numbers[-1])
		unit_numbers = np.array(unit_numbers)
		max_unit = np.max(unit_numbers)
		unit_name = 'unit%03d' % int(max_unit + 1)

	# Get a new unit_descriptor table row for this new unit
	unit_description = table.row	

	# Load data from the chosen electrode and solution
	spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' % electrode_num)
	spike_times = np.load('./spike_times/electrode%i/spike_times.npy' % electrode_num)
	predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' % (electrode_num, num_clusters))
	
	# If only 1 cluster was chosen, add that as a new unit in /sorted_units. Ask if the isolated unit is an almost-SURE single unit
	if len(clusters) == 1:
		hf5.createGroup('/sorted_units', unit_name)
		unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
		unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
		waveforms = hf5.createArray('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
		times = hf5.createArray('/sorted_units/%s' % unit_name, 'times', unit_times)
		unit_description['electrode_number'] = electrode_num
		single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
		unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
		unit_description.append()
		table.flush()
		hf5.flush()

	else:
		# If the chosen units are going to be merged, merge them
		if merge:
			unit_waveforms = []
			unit_times = []
			for cluster in clusters:
				if unit_waveforms == []:
					unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]			
					unit_times = spike_times[np.where(predictions == int(cluster))[0]]
				else:
					unit_waveforms = np.concatenate((unit_waveforms, spike_waveforms[np.where(predictions == int(cluster))[0], :]))
					unit_times = np.concatenate((unit_times, spike_times[np.where(predictions == int(cluster))[0]]))
			# Warn the user about the frequency of ISI violations in the merged unit
			ISIs = np.ediff1d(unit_times)/30.0
			violations = np.where(ISIs < 2.0)[0]
			proceed = easygui.multchoicebox(msg = 'My merged cluster has %f percent (%i/%i) ISI violations (<2ms). I want to still merge these clusters into one unit (True = Yes, False = No)' % ((float(len(violations))/float(len(unit_times)))*100.0, len(violations), len(unit_times)), choices = ('True', 'False'))
			proceed = ast.literal_eval(proceed[0])

			# Create unit if the user agrees to proceed, else abort and go back to start of the loop 
			if proceed:	
				hf5.createGroup('/sorted_units', unit_name)
				waveforms = hf5.createArray('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
				times = hf5.createArray('/sorted_units/%s' % unit_name, 'times', unit_times)
				unit_description['electrode_number'] = electrode_num
				single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
				unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
				unit_description.append()
				table.flush()
				hf5.flush()
			else:
				continue

		# Otherwise include each cluster as a separate unit
		else:
			for cluster in clusters:
				hf5.createGroup('/sorted_units', unit_name)
				unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
				unit_times = spike_times[np.where(predictions == int(cluster))[0]]
				waveforms = hf5.createArray('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
				times = hf5.createArray('/sorted_units/%s' % unit_name, 'times', unit_times)
				unit_description['electrode_number'] = electrode_num
				single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster)), choices = ('True', 'False'))
				unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
				unit_description.append()
				table.flush()
				hf5.flush()				

				# Finally increment max_unit and create a new unit name
				max_unit += 1
				unit_name = 'unit%03d' % int(max_unit + 1)

				# Get a new unit_descriptor table row for this new unit
				unit_description = table.row

# Close the hdf5 file
hf5.close()
	 



	




