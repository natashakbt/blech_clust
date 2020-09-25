
# Import stuff!
import os
import shutil
import easygui
import sys
import glob

# Get name of directory with the data files
if sys.argv[1] != '':
    dir_name = os.path.abspath(sys.argv[1])
else:
    dir_name = easygui.diropenbox('Please select data directory')

file_list = os.listdir(dir_name)

# Mark for removal:
removal_files = ['*.h5','*.params','results.log']
file_paths = [glob.glob(os.path.join(dir_name, x)) for x in removal_files]
file_paths = [x[0] for x in file_paths if len(x) > 0]
removal_dirs = ['clustering_results', 'memory_monitor_clustering',
        'Plots','spike_times','spike_waveforms']
dir_paths = [os.path.join(dir_name,x) for x in removal_dirs]
for this_file in file_paths:
    try:
        os.remove(this_file)
    except:
        pass
for this_dir in dir_paths:
    try:
        shutil.rmtree(this_dir)
    except:
        pass

