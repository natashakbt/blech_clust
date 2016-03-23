# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os

# Ask for the directory where the hdf5 file sits
dir_name = easygui.diropenbox()

# Store the directory path to blech.dir
f = open('blech.dir', 'w')
print >>f, dir_name
f.close()

# Change to the directory
os.chdir(dir_name)

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files

'''# Delete the HMM_plots folder if it exists
try:
	os.system("rm -r ./HMM_plots")
except:
	pass'''

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r')

# Ask the user for the HMM parameters  
hmm_params = easygui.multenterbox(msg = 'Fill in the parameters for running a HMM (Poisson or Multinomial emissions) on your data', fields = ['Minimum number of states', 'Maximum number of states', 'Maximum number of iterations (300 is more than enough)', 'Convergence criterion (usually 1e-9)', 'Number of random restarts for HMM (50-60 is more than enough)', 'Transition probability inertia (between 0 and 1)', 'Emission Distribution intertia (between 0 and 1)'])

# Ask the user for the taste to run the HMM on
tastes = hf5.list_nodes('/spike_trains')
hmm_taste = easygui.multchoicebox(msg = 'Which taste do you want to run the HMM on?', choices = ([str(taste).split('/')[-1] for taste in tastes]))
taste_num = 0
for i in range(len(tastes)):
	if str(tastes[i]).split('/')[-1] in hmm_taste:
		taste_num = i

# Create the folder for storing the plots coming in from HMM analysis of the data - pass if it exists
try:
	os.mkdir("HMM_plots")
	# Make folders for storing plots from each of the tastes within HMM_plots
	for i in range(len(tastes)):
		os.mkdir('HMM_plots/dig_in_%i' % i)
except: 
	pass


# Ask the user for the parameters to process spike trains
spike_params = easygui.multenterbox(msg = 'Fill in the parameters for processing your spike trains', fields = ['Pre-stimulus time used for making spike trains (ms)', 'Bin size for HMM (ms) - usually 10', 'Pre-stimulus time for HMM (ms)', 'Post-stimulus time for HMM (ms)'])

# Print the paramaters to blech.hmm_params
f = open('blech.hmm_params', 'w')
for params in hmm_params:
	print>>f, params
print>>f, taste_num
for params in spike_params:
	print>>f, params
f.close()

# Grab Brandeis unet username
username = easygui.multenterbox(msg = 'Enter your Brandeis unet id', fields = ['unet username'])

# Dump shell file for running parallel job on the user's blech_clust folder on the desktop
os.chdir('/home/%s/Desktop/blech_clust' % username[0])
f = open('blech_multinomial_hmm.sh', 'w')
g = open('blech_poisson_hmm.sh', 'w')
print >>f, "module load PYTHON/ANACONDA-1.8.0"
print >>g, "module load PYTHON/ANACONDA-1.8.0"
print >>f, "export PYTHONPATH=/share/apps/scisoft/PYTHON-MODULES/ANACONDA-1.8.0/lib/python2.7/site-packages/:$PYTHONPATH"
print >>g, "export PYTHONPATH=/share/apps/scisoft/PYTHON-MODULES/ANACONDA-1.8.0/lib/python2.7/site-packages/:$PYTHONPATH"
print >>f, "cd /home/%s/Desktop/blech_clust" % username[0]
print >>g, "cd /home/%s/Desktop/blech_clust" % username[0]
print >>f, "python blech_multinomial_hmm.py"
print >>g, "python blech_poisson_hmm.py"
f.close()
g.close()

hf5.close()



