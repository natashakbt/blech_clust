# Import stuff!
import numpy as np
import tables
import pylab as plt
import easygui
import sys
import os
import json
import glob

# Ask for the directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
if len(sys.argv) > 1:
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')
os.chdir(dir_name)

# Make directory to store the PSTH plots. Delete and remake the directory if it exists
try:
	os.system('rm -r '+'./overlay_PSTH')
except:
	pass
os.mkdir('./overlay_PSTH')

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Params file with processing paramters
json_name = glob.glob('./**.params')[0]
with open(json_name,'r') as params_file_connect:
    params_dict = json.load(params_file_connect)

# Json file with EXPERIMENTAL parameters
dir_basename = os.path.basename(dir_name[:-1])
json_path = glob.glob(os.path.join(dir_name, dir_basename + '.info'))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)

# Now ask the user to put in the identities of the digital inputs
trains_dig_in = hf5.list_nodes('/spike_trains')
# Pull identities from the json file
identities = info_dict['taste_params']['tastes'] 
#identities = easygui.multenterbox(msg = 'Put in the taste identities of the digital inputs', fields = [train._v_name for train in trains_dig_in])

# Plot all tastes
plot_tastes_dig_in = np.arange(len(identities))

# Ask what taste to plot
#plot_tastes = easygui.multchoicebox(msg = 'Which tastes do you want to plot? ', choices = ([taste for taste in identities]))
#plot_tastes_dig_in = []
#for taste in plot_tastes:
#    plot_tastes_dig_in.append(identities.index(taste))

# Ask the user for the pre stimulus duration used while making the spike arrays
#pre_stim = easygui.multenterbox(msg = 'What was the pre-stimulus duration pulled into the spike arrays?', fields = ['Pre stimulus (ms)'])
#pre_stim = int(pre_stim[0])
pre_stim = params_dict['spike_array_durations'][0]

# Get the psth paramaters from the user
#params = easygui.multenterbox(msg = 'Enter the parameters for making the PSTHs', fields = ['Window size (ms)', 'Step size (ms)'])
#for i in range(len(params)):
#	params[i] = int(params[i])
params = [params_dict['psth_params']['window_size'], 
            params_dict['psth_params']['step_size']]

# Ask the user about the type of units they want to do the calculations on (single or all units)
#chosen_units = []
all_units = np.arange(trains_dig_in[0].spike_array.shape[1])
#chosen_units = easygui.multchoicebox(msg = 'Which units do you want to choose? (*same as in setup*)', choices = ([i for i in all_units]))
#for i in range(len(chosen_units)):
#	chosen_units[i] = int(chosen_units[i])
#chosen_units = np.array(chosen_units)
chosen_units = all_units

# Extract neural response data from hdf5 file
response = hf5.root.ancillary_analysis.scaled_neural_response[:]
trial_count = int(response.shape[2]/len(trains_dig_in))
num_units = len(chosen_units)
num_tastes = len(trains_dig_in)
x = np.arange(0, 6751, params[1]) - 2000
plot_places = np.where((x>=-1000)*(x<=4000))[0]

for i in range(num_units):
    fig = plt.figure(figsize=(18, 6))

    # First plot
    plt.subplot(121)
    plt.title('Unit: %i, Window size: %i ms, Step size: %i ms' % \
            (chosen_units[i], params[0], params[1]))
    for j in plot_tastes_dig_in:
            plt.plot(x[plot_places], 
                    1000*np.mean(response[plot_places, i, trial_count*j:trial_count*(j+1)], 
                        axis = 1), label = identities[j])
    plt.legend()
    plt.xlabel('Time from taste delivery (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.legend(loc='upper left', fontsize=10)

    # Second plot
    plt.subplot(122)
    exec('waveforms = hf5.root.sorted_units.unit%03d.waveforms[:]' % (chosen_units[i]))
    t = np.arange(waveforms.shape[1])
    plt.plot(t - 15, waveforms.T, linewidth = 0.01, color = 'red')
    plt.xlabel('Time (samples (30 per ms))')
    plt.ylabel('Voltage (microvolts)')
    title_str = f"Unit {chosen_units[i]}," \
                f"Total waveforms = {waveforms.shape[0]}\n"\
                f"Electrode: {hf5.root.unit_descriptor[chosen_units[i]]['electrode_number']},"\
                f"Single Unit: {hf5.root.unit_descriptor[chosen_units[i]]['single_unit']},"\
                f"RSU: {hf5.root.unit_descriptor[chosen_units[i]]['regular_spiking']},"\
                f"FS: {hf5.root.unit_descriptor[chosen_units[i]]['fast_spiking']}"
    plt.title(title_str)
    fig.savefig('./overlay_PSTH/' + '/Unit%i.png' % (chosen_units[i]), bbox_inches = 'tight')
    plt.close("all")
    print(f'Completed Unit {i}')

# Close hdf5 file
hf5.close()



