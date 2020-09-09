import os
import tables
import numpy as np
import easygui
import ast
import sys
import re
import pylab as plt
import matplotlib.image as mpimg
from sklearn.mixture import GaussianMixture
import blech_waveforms_datashader

# Get directory where the hdf5 file sits, and change to that directory
# Get name of directory with the data files
if sys.argv[1] != '':
    dir_name = os.path.abspath(sys.argv[1])
    if dir_name[-1] != '/':
        dir_name += '/'
else:
    dir_name = easygui.diropenbox(msg = 'Please select data directory')

#dir_name = easygui.diropenbox()
os.chdir(dir_name)

# Clean up the memory monitor files, pass if clean up has been done already
if not os.path.exists('./memory_monitor_clustering/memory_usage.txt'):
        file_list = os.listdir('./memory_monitor_clustering')
        f = open('./memory_monitor_clustering/memory_usage.txt', 'w')
        for files in file_list:
                try:
                        mem_usage = np.loadtxt('./memory_monitor_clustering/' + files)
                        print('electrode'+files[:-4], '\t', str(mem_usage)+'MB', file=f)
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
hf5 = tables.open_file(hdf5_name, 'r+')

# Delete the raw node, if it exists in the hdf5 file, to cut down on file size
try:
        hf5.remove_node('/raw', recursive = 1)
        # And if successful, close the currently open hdf5 file and ptrepack the file
        hf5.close()
        print("Raw recordings removed")
        os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 "
            "--complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
        # Delete the old (raw and big) hdf5 file
        os.system("rm " + hdf5_name)
        # And open the new, repacked file
        hf5 = tables.open_file(hdf5_name[:-3] + "_repacked.h5", 'r+')
        print("File repacked")
except:
        print("Raw recordings have already been removed, so moving on ..")

# Make the sorted_units group in the hdf5 file if it doesn't already exist
try:
        hf5.create_group('/', 'sorted_units')
except:
        pass

# Define a unit_descriptor class to be used to add things (anything!) 
# about the sorted units to a pytables table
class unit_descriptor(tables.IsDescription):
        electrode_number = tables.Int32Col()
        single_unit = tables.Int32Col()
        regular_spiking = tables.Int32Col()
        fast_spiking = tables.Int32Col()
        waveform_count = tables.Int32Col()

# Make a table under /sorted_units describing the sorted units. 
# If unit_descriptor already exists, just open it up in the variable table
try:
        table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
except:
        table = hf5.root.unit_descriptor

def entry_checker(msg, check_func, fail_response):
    check_bool = False
    continue_bool = True
    exit_str = '"x" to exit :: '
    while not check_bool:
        msg_input = input(msg.join([' ',exit_str]))
        if msg_input == 'x':
            continue_bool = False
            break
        check_bool = check_func(msg_input)
        if not check_bool:
            print(fail_response)
    return msg_input, continue_bool

# Run an infinite loop as long as the user wants to pick clusters from the electrodes   
while True:

        # Get electrode number from user
        electrode_num_str, continue_bool = entry_checker(\
                msg = 'Please enter electrode number :: ',
                check_func = str.isdigit,
                fail_response = 'Please enter an interger')

        if continue_bool:
            electrode_num = int(electrode_num_str)
        else:
            break

        #electrode_num = input('Please enter electrode number :: ')
        # Break if wrong input/cancel command was given
        #try:
        #        electrode_num = int(electrode_num[0])
        #except:
        #        break
        #
        # Get the number of clusters in the chosen solution

        
        #num_clusters = input('Please enter solution number :: ')
        num_clusters_str, continue_bool = entry_checker(\
                msg = 'Please enter solution number :: ',
                check_func = lambda x: (str.isdigit(x) and (1<int(x)<=7)),
                fail_response = 'Please enter an interger')
        if continue_bool:
            num_clusters = int(num_clusters_str)
        else:
            break

        #num_clusters = easygui.multenterbox(\
        #        msg = 'Which solution do you want to choose for electrode %i?' % electrode_num, 
        #       fields = ['Number of clusters in the solution'])
        #num_clusters = int(num_clusters[0])

        print('Loading data for solution')
        # Load data from the chosen electrode and solution
        spike_waveforms = np.load('./spike_waveforms/electrode%i/spike_waveforms.npy' \
                % electrode_num)
        spike_times = np.load('./spike_times/electrode%i/spike_times.npy' % electrode_num)
        pca_slices = np.load('./spike_waveforms/electrode%i/pca_waveforms.npy' % electrode_num)
        energy = np.load('./spike_waveforms/electrode%i/energy.npy' % electrode_num)
        amplitudes = np.load('./spike_waveforms/electrode%i/spike_amplitudes.npy' % electrode_num)
        predictions = np.load('./clustering_results/electrode%i/clusters%i/predictions.npy' \
                % (electrode_num, num_clusters))

        # Get cluster choices from the chosen solution
        #clusters = easygui.multchoicebox(msg = 'Which clusters do you want to choose?', \
                # choices = tuple([str(i) for i in range(int(np.max(predictions) + 1))]))

        def cluster_check(x):
            clusters = re.findall('[0-9]+',x)
            return sum([i.isdigit() for i in clusters]) == len(clusters)

        clusters_msg, continue_bool = entry_checker(\
                msg = 'Please enter cluster numbers (anything separated) ::',
                check_func = cluster_check,
                fail_response = 'Please enter integers')
        if continue_bool:
            clusters = re.findall('[0-9]+',clusters_msg)
            clusters = [int(x) for x in clusters]
        else:
            break
        
        # Print out selections
        print('Electrode {}, Solution {}, Cluster {}'.\
                format(electrode_num, num_clusters, clusters))

        # Re-show images of neurons so dumb people like Abu can make sure they
        # picked the right ones
        fig, ax = plt.subplots(len(clusters), 2)
        for cluster_num, cluster in enumerate(clusters):
            isi_plot = mpimg.imread(
                                    './Plots/{}/{}_clusters_waveforms_ISIs/'\
                                    'Cluster{}_ISIs.png'\
                                    .format(electrode_num, num_clusters, cluster)) 
            waveform_plot =  mpimg.imread(
                                    './Plots/{}/{}_clusters_waveforms_ISIs/'\
                                    'Cluster{}_waveforms.png'\
                                    .format(electrode_num, num_clusters, cluster)) 
            if len(clusters) < 2:
                ax[0].imshow(isi_plot,aspect='auto');ax[0].axis('off')
                ax[1].imshow(waveform_plot,aspect='auto');ax[1].axis('off')
            else:
                ax[cluster_num, 0].imshow(isi_plot,aspect='auto');ax[cluster_num,0].axis('off')
                ax[cluster_num, 1].imshow(waveform_plot,aspect='auto');ax[cluster_num,1].axis('off')
        fig.suptitle('Are these the neurons you want to select?')
        fig.tight_layout()
        plt.show()

        # Check if the user wants to merge clusters if more than 1 cluster was chosen. 
        # Else ask if the user wants to split/re-cluster the chosen cluster
        merge = False
        re_cluster = False
        merge_msg = 'a'
        re_cluster_msg = 'a'
        if len(clusters) > 1:
            merge_msg, continue_bool = entry_checker(\
                    msg = 'MERGE all clusters? (y/n)',
                    check_func = lambda x: x in ['y','n'],
                    fail_response = 'Please enter (y/n)')
            if continue_bool:
                if merge_msg == 'y': 
                    merge = True
                elif merge_msg == 'n': 
                    merge = False
            else:
                break

            #while merge_msg not in ['y','n']: 
            #    merge_msg = input('I want to MERGE these clusters (y/n) :: ')
            #    if merge_msg == 'y': 
            #        merge = True
            #    elif merge_msg == 'n': 
            #        merge = False
            #    else:
            #        print('Please enter a valid option')
                #merge = easygui.multchoicebox(msg = 'I want to merge these clusters into one unit (True = Yes, False = No)', choices = ('True', 'False'))
                #merge = ast.literal_eval(merge[0])
        else:
            split_msg, continue_bool = entry_checker(\
                    msg = 'SPLIT this cluster? (y/n)',
                    check_func = lambda x: x in ['y','n'],
                    fail_response = 'Please enter (y/n)')
            if continue_bool:
                if split_msg == 'y': 
                    re_cluster = True
                elif split_msg == 'n': 
                    re_cluster = False
            else:
                break

            #while merge_msg not in ['y','n']: 
            #    re_cluster_msg = input('I want to SPLIT these clusters (y/n) :: ')
            #    if re_cluster_msg == 'y': 
            #        re_cluster = True
            #    elif re_cluster_msg == 'n': 
            #        re_cluster = False
            #    else:
            #        print('Please enter a valid option')
                #re_cluster = easygui.multchoicebox(msg = 'I want to split this cluster (True = Yes, False = No)', choices = ('True', 'False'))
                #re_cluster = ast.literal_eval(re_cluster[0])

        # If the user asked to split/re-cluster, ask them for the clustering parameters and perform clustering
        split_predictions = []
        chosen_split = 0
        if re_cluster: 
                # Get clustering parameters from user
                n_clusters = int(input('Number of clusters (default=5): ') or "5")
                fields = ['Maximum number of iterations (1000 is more than enough)', 
                        'Convergence criterion (usually 0.0001)', 
                        'Number of random restarts for GMM (10 is more than enough)']
                values = [100,0.001,10]
                for x in zip(fields, values):
                    print(str(x)) 
                edit_bool = 'a'
                edit_bool_msg, continue_bool = entry_checker(\
                        msg = 'Use these parameters? (y/n)',
                        check_func = lambda x: x in ['y','n'],
                        fail_response = 'Please enter (y/n)')
                if continue_bool:
                    if edit_bool_msg == 'n': 
                        clustering_params = easygui.multenterbox(msg = 'Fill in the'\
                                'parameters for re-clustering (using a GMM)', 
                                fields  = fields, values = values)
                        n_iter = int(clustering_params[1])
                        thresh = float(clustering_params[2])
                        n_restarts = int(clustering_params[3]) 
                else:
                    break
                #while edit_bool not in ['y','n']:
                #    edit_bool = input('Use these parameters? (y/n) :: ')
                #if edit_bool == 'n':
                #    clustering_params = easygui.multenterbox(msg = 'Fill in the'\
                #            'parameters for re-clustering (using a GMM)', 
                #            fields  = fields, values = values)
                #    n_iter = int(clustering_params[1])
                #    thresh = float(clustering_params[2])
                #    n_restarts = int(clustering_params[3]) 

                # Make data array to be put through the GMM - 5 components: 
                # 3 PCs, scaled energy, amplitude
                this_cluster = np.where(predictions == int(clusters[0]))[0]
                n_pc = 3
                data = np.zeros((len(this_cluster), n_pc + 2))  
                data[:,2:] = pca_slices[this_cluster,:n_pc]
                data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
                data[:,1] = np.abs(amplitudes[this_cluster])/np.max(np.abs(amplitudes[this_cluster]))

                # Cluster the data
                g = GaussianMixture(
                        n_components = n_clusters, 
                        covariance_type = 'full', 
                        tol = thresh, 
                        max_iter = n_iter, 
                        n_init = n_restarts)
                g.fit(data)
        
                # Show the cluster plots if the solution converged
                if g.converged_:
                        split_predictions = g.predict(data)
                        x = np.arange(len(spike_waveforms[0])/10) + 1
                        for cluster in range(n_clusters):
                                split_points = np.where(split_predictions == cluster)[0]                                
                                # plt.figure(cluster)
                                # Waveforms and times from the chosen cluster
                                slices_dejittered = spike_waveforms[this_cluster, :]            
                                times_dejittered = spike_times[this_cluster]
                                # Waveforms and times from the chosen split of the chosen cluster
                                times_dejittered = times_dejittered[split_points]               
                                ISIs = np.ediff1d(np.sort(times_dejittered))/30.0
                                violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                                violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                                fig, ax = blech_waveforms_datashader.waveforms_datashader(\
                                        slices_dejittered[split_points, :], x)
                                # plt.plot(x-15, slices_dejittered[split_points, :].T, 
                                #                   linewidth = 0.01, color = 'red')
                                ax.set_xlabel('Sample (30 samples per ms)')
                                ax.set_ylabel('Voltage (microvolts)')
                                print_str = (f'Split cluster {cluster} '
                                    f'has {violations2:.1f} % (<2ms) '
                                    f'and {violations1:.1f} % (<1ms) ISI out of '
                                    f'{split_points.shape[0]} total waveforms. \n') 
                                #ax.set_title(\
                                #        "Split Cluster{:d}, 2ms ISI violations={:.1f} percent".\
                                #        format(cluster, violations2) + "\n" + \
                                #        "1ms ISI violations={:.1f}%, \
                                #        Number of waveforms={:d}".\
                                #        format(violations1, split_points.shape[0]))
                else:
                        print("Solution did not converge "\
                                "- try again with higher number of iterations "\
                                "or lower convergence criterion")
                        continue

                plt.show()
                # Ask the user for the split clusters they want to choose
                choice_list = tuple([str(i) for i in range(n_clusters)]) 

                clusters_msg, continue_bool = entry_checker(\
                        msg = f'Please select from {choice_list} (anything separated) ::',
                        check_func = cluster_check,
                        fail_response = 'Please enter integers')
                if continue_bool:
                    clusters = re.findall('[0-9]+',clusters_msg)
                    clusters = [int(x) for x in clusters]
                else:
                    break

                #check_bool = 0
                #while not check_bool:
                #    cluster_msg = input(f'Please enter cluster numbers (anything'
                #            'separated) \n {choice_list} \n :: ')
                #    clusters = re.findall('[0-9]+',cluster_msg)
                #    if sum([(x in choice_list) for x in clusters]) == len(clusters):
                #        check_bool = 1
                #clusters = [int(x) for x in clusters]

                #chosen_split = easygui.multchoicebox(msg = 'Which split cluster'\
                #        ' do you want to choose? Hit cancel to exit', 
                #        choices = choice_list)
                try:
                        #chosen_split = int(chosen_split[0])
                        chosen_split = [int(num) for num,val in \
                                enumerate(choice_list) if val in chosen_split]
                        print('Selected split clusters: {}'.format(chosen_split))
                except:
                        continue

        # Get list of existing nodes/groups under /sorted_units
        node_list = hf5.list_nodes('/sorted_units')

        # If node_list is empty, start naming units from 000
        unit_name = ''
        max_unit = 0
        if node_list == []:             
                unit_name = 'unit%03d' % 0
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

        # If the user re-clustered/split clusters, add the chosen clusters in split_clusters
        if re_cluster:
                hf5.create_group('/sorted_units', unit_name)
                # Waveforms of originally chosen cluster
                unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]    
                unit_waveforms = np.concatenate(\
                        [unit_waveforms[np.where(split_predictions == this_split)[0], :] \
                                    for this_split in chosen_split])
                # Subsetting this set of waveforms to include only the chosen split
                # Do the same thing for the spike times
                unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
                unit_times = np.concatenate(\
                        [unit_times[np.where(split_predictions == this_split)[0]] \
                                    for this_split in chosen_split])
                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 
                                'waveforms', unit_waveforms)
                times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                unit_description['waveform_count'] = int(len(unit_times))
                unit_description['electrode_number'] = electrode_num

                single_unit_msg, continue_bool = entry_checker(\
                        msg = 'Single-unit? (y/n)',
                        check_func = lambda x: x in ['y','n'],
                        fail_response = 'Please enter (y/n)')
                if continue_bool:
                    if single_unit_msg == 'y': 
                        single_unit = True
                    elif single_unit_msg == 'n': 
                        single_unit = False
                else:
                    break

                #single_unit_msg = 'a'
                #while single_unit_msg not in ['y','n']:
                #    single_unit_msg = input('Single-unit? (y/n) : ')
                #if single_unit_msg == 'y':
                #    single_unit = True 
                #else:
                #    single_unit = False
                #single_unit = easygui.multchoicebox(\
                #        msg = 'I am almost-SURE that this is a beautiful single', 
                #        'unit (True = Yes, False = No)', 
                #        choices = ('True', 'False'))
                #unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                unit_description['single_unit'] = int(single_unit)
                # If the user says that this is a single unit, 
                # ask them whether its regular or fast spiking
                unit_description['regular_spiking'] = 0
                unit_description['fast_spiking'] = 0

                if single_unit:
                    unit_type_msg, continue_bool = entry_checker(\
                            msg = 'Regular or fast spiking? (r/f)',
                            check_func = lambda x: x in ['r','f'],
                            fail_response = 'Please enter (r/f)')
                    if continue_bool:
                        if unit_type_msg == 'r': 
                            unit_type = 'regular_spiking'
                        elif unit_type_msg == 'f': 
                            unit_type = 'fast_spiking'
                    else:
                        break

                    unit_description[unit_type[0]] = 1              
                    unit_description.append()

                #if single_unit:
                #    unit_type_msg = 'a'
                #    while unit_type_msg not in ['r','f']:
                #        unit_type_msg = input('Unit type? (r/f) : ')
                #    if unit_type_msg == 'r':
                #        unit_type = 'regular_spiking'
                #    else:
                #        unit_type = 'fast_spiking'
                #if int(ast.literal_eval(single_unit[0])):
                        #unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                #unit_description[unit_type[0]] = 1              
                table.flush()
                hf5.flush()
                

        # If only 1 cluster was chosen (and it wasn't split), add that as a new unit in /sorted_units. Ask if the isolated unit is an almost-SURE single unit
        elif len(clusters) == 1:
                hf5.create_group('/sorted_units', unit_name)
                unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
                unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                unit_description['waveform_count'] = int(len(unit_times))
                unit_description['electrode_number'] = electrode_num
                single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
                unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                # If the user says that this is a single unit, ask them whether its regular or fast spiking
                unit_description['regular_spiking'] = 0
                unit_description['fast_spiking'] = 0
                if single_unit:
                    unit_type_msg, continue_bool = entry_checker(\
                            msg = 'Regular or fast spiking? (r/f)',
                            check_func = lambda x: x in ['r','f'],
                            fail_response = 'Please enter (r/f)')
                    if continue_bool:
                        if unit_type_msg == 'r': 
                            unit_type = 'regular_spiking'
                        elif unit_type_msg == 'f': 
                            unit_type = 'fast_spiking'
                    else:
                        break

                    #unit_type_msg = 'a'
                    #while unit_type_msg not in ['r','f']:
                    #    unit_type_msg = input('Unit type? (r/f) : ')
                    #if unit_type_msg == 'r':
                    #    unit_type = 'regular_spiking'
                    #else:
                        unit_type = 'fast_spiking'
                #if int(ast.literal_eval(single_unit[0])):
                        #unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                        #unit_description[unit_type[0]] = 1              
                    unit_description[unit_type[0]] = 1              
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

                        # Show the merged cluster to the user, and ask if they still want to merge
                        x = np.arange(len(unit_waveforms[0])/10) + 1
                        fig, ax = blech_waveforms_datashader.waveforms_datashader(unit_waveforms, x)
                        # plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
                        ax.set_xlabel('Sample (30 samples per ms)')
                        ax.set_ylabel('Voltage (microvolts)')
                        ax.set_title('Merged cluster, No. of waveforms={:d}'.format(unit_waveforms.shape[0]))
                        plt.show()
 
                        # Warn the user about the frequency of ISI violations in the merged unit
                        ISIs = np.ediff1d(np.sort(unit_times))/30.0
                        violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
                        violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
                        print_str = (f'My merged cluster has {violations1:.1f} % (<2ms) '
                            f'and {violations2:.1f} % (<1ms) ISI out of '
                            f'{len(unit_times)} total waveforms. \n' 
                            'I want to still merge these clusters into one unit (y/n) :: ')
                        proceed_msg, continue_bool = entry_checker(\
                                msg = print_str, 
                                check_func = lambda x: x in ['y','n'],
                                fail_response = 'Please enter (y/n)')
                        if continue_bool:
                            if proceed_msg == 'y': 
                                proceed = True
                            elif proceed_msg == 'n': 
                                proceed = False
                        else:
                            break
                        #proceed_msg = 'a'
                        #while proceed_msg not in ['y','n']:
                        #    proceed_msg = input(print_str)
                        #if proceed_msg == 'y':
                        #    proceed = True
                        #else:
                        #    proceed = False
                        #proceed = easygui.multchoicebox(msg=print_str, choices = ('True', 'False'))
                        #proceed = ast.literal_eval(proceed[0])

                        # Create unit if the user agrees to proceed, else abort and go back to start of the loop 
                        if proceed:     
                                hf5.create_group('/sorted_units', unit_name)
                                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                                times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                                unit_description['waveform_count'] = int(len(unit_times))
                                unit_description['electrode_number'] = electrode_num

                                if single_unit:
                                single_unit_msg, continue_bool = entry_checker(\
                                        msg = 'Single-unit? (y/n)',
                                        check_func = lambda x: x in ['y','n'],
                                        fail_response = 'Please enter (y/n)')
                                if continue_bool:
                                    if single_unit_msg == 'y': 
                                        single_unit = True
                                    elif single_unit_msg == 'n': 
                                        single_unit = False
                                else:
                                    break

                                #single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that this is a beautiful single unit (True = Yes, False = No)', choices = ('True', 'False'))
                                #unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                                # If the user says that this is a single unit, ask them whether its regular or fast spiking
                                unit_description['regular_spiking'] = 0
                                unit_description['fast_spiking'] = 0
                                if single_unit
                                    unit_type_msg, continue_bool = entry_checker(\
                                            msg = 'Regular or fast spiking? (r/f)',
                                            check_func = lambda x: x in ['r','f'],
                                            fail_response = 'Please enter (r/f)')
                                    if continue_bool:
                                        if unit_type_msg == 'r': 
                                            unit_type = 'regular_spiking'
                                        elif unit_type_msg == 'f': 
                                            unit_type = 'fast_spiking'
                                    else:
                                        break

                                #if int(ast.literal_eval(single_unit[0])):
                                #        unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                                        unit_description[unit_type] = 1
                                        #unit_description[unit_type[0]] = 1
                                unit_description.append()
                                table.flush()
                                hf5.flush()
                        else:
                                continue

                # Otherwise include each cluster as a separate unit
                else:
                        for cluster in clusters:
                                hf5.create_group('/sorted_units', unit_name)
                                unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
                                unit_times = spike_times[np.where(predictions == int(cluster))[0]]
                                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                                times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                                unit_description['waveform_count'] = int(len(unit_times))
                                unit_description['electrode_number'] = electrode_num

                                single_unit_msg, continue_bool = entry_checker(\
                                        msg = 'Single-unit? (y/n)',
                                        check_func = lambda x: x in ['y','n'],
                                        fail_response = 'Please enter (y/n)')
                                if continue_bool:
                                    if single_unit_msg == 'y': 
                                        single_unit = True
                                    elif single_unit_msg == 'n': 
                                        single_unit = False
                                else:
                                    break

                                #single_unit = easygui.multchoicebox(msg = 'I am almost-SURE that electrode: %i cluster: %i is a beautiful single unit (True = Yes, False = No)' % (electrode_num, int(cluster)), choices = ('True', 'False'))

                                unit_description['single_unit'] = int(single_unit)
                                #unit_description['single_unit'] = int(ast.literal_eval(single_unit[0]))
                                # If the user says that this is a single unit, ask them whether its regular or fast spiking
                                unit_description['regular_spiking'] = 0
                                unit_description['fast_spiking'] = 0

                                if single_unit:
                                    unit_type_msg, continue_bool = entry_checker(\
                                            msg = 'Regular or fast spiking? (r/f)',
                                            check_func = lambda x: x in ['r','f'],
                                            fail_response = 'Please enter (r/f)')
                                    if continue_bool:
                                        if unit_type_msg == 'r': 
                                            unit_type = 'regular_spiking'
                                        elif unit_type_msg == 'f': 
                                            unit_type = 'fast_spiking'
                                    else:
                                        break

                                #if int(ast.literal_eval(single_unit[0])):
                                #        unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                                        unit_description[unit_type] = 1
                                        #unit_description[unit_type[0]] = 1
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
         



        




