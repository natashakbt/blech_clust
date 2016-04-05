# Sets up emg data for running the envelope of emg recordings (env.npy) through a local Bayesian Spectrum Analysis (BSA). 
# Needs an installation of R (installing Rstudio on Ubuntu is enough) - in addition, the R library BaSAR needs to be installed from the CRAN archives (https://cran.r-project.org/src/contrib/Archive/BaSAR/)
# This is the starting step for emg_local_BSA_execute.py

# Import stuff
import numpy as np
import easygui
import os

# Change to the directory that has the emg data files (env.npy and sig_trials.npy). Make a directory for storing the BSA results and change to it
dir_name = easygui.diropenbox()
os.chdir(dir_name)
os.makedirs('emg_BSA_results')

# Load the data files
env = np.load('env.npy')
sig_trials = np.load('sig_trials.npy')

# Ask for the HPC queue to use
queue = easygui.multchoicebox(msg = 'Which HPC queue do you want to use for EMG analysis?', choices = ('neuro.q', 'dk.q'))

# Grab Brandeis unet username
username = easygui.multenterbox(msg = 'Enter your Brandeis unet id', fields = ['unet username'])

# Dump a shell file for the BSA analysis in the user's blech_clust directory on the desktop
os.chdir('/home/%s/Desktop/blech_clust' % username[0])
f = open('blech_emg.sh', 'w')
print >>f, "module load PYTHON/ANACONDA-2.5.0"
print >>f, "module load R"
print >>f, "cd /home/%s/Desktop/blech_clust" % username[0]
print >>f, "python emg_local_BSA_execute.py"
f.close()

print "Now logout of the compute node and go back to the login node. Then say: qsub -t 1-"+str(sig_trials.shape[0]*sig_trials.shape[1])+" -q "+queue[0]+" blech_emg.sh"







