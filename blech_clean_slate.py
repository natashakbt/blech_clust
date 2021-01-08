
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

# Keep certain files and remove everything else
keep_pattern = ['*.dat','*.info','*.rhd', '*.csv']
keep_files = []
for pattern in keep_pattern:
    keep_files.extend(glob.glob(os.path.join(dir_name, pattern)))
keep_files_basenames = [os.path.basename(x) for x in keep_files]

remove_files = [x for x in file_list if x not in keep_files_basenames] 
remove_paths = [os.path.join(dir_name,x) for x in remove_files]

for x in remove_paths:
    try:
        shutil.rmtree(x)
    except:
        os.remove(x)
    finally:
        pass

