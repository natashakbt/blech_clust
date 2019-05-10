# Import stuff!
import easygui
import sys
import os

# Ask for the directory where the first dataset sits
dir_name1 = easygui.diropenbox(msg = 'Where is the data from the first session?', title = 'First session of data')
# Change to that directory
os.chdir(dir_name1)
# Get the list of filenames (only the Intan .dat files)
files1 = [filename for filename in os.listdir(dir_name1) if filename[-4:] == ".dat"]

# Now do the same for the second session of data
dir_name2 = easygui.diropenbox(msg = 'Where is the data from the second session?', title = 'Second session of data')

# Get the output directory for the joined files
dir_output = easygui.diropenbox(msg = 'Where do you want to save the joined files?', title = 'Output directory')

# Read through the first set of files, append the second set and save to the output directory
for file1 in files1:
	try:
		os.system("cat " + dir_name1 + "/" + file1 + " " + dir_name2 + "/" + file1 + " > " + dir_output + "/" + file1)
	except Exception:
		continue

# Copy one of the info.rhd files to the output folder
os.system("cp " + dir_name1 + "/info.rhd " + dir_output)




