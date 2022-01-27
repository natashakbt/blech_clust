"""
Generates plots in data directory for entire timeseries of 
DIG_INs and AMP channels
"""

import argparse
import glob
import os

import easygui
import numpy as np
import pylab as plt
from tqdm import tqdm

# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(description = 'Plots DIG_INs and AMP files')
parser.add_argument('dir_name',  help = 'Directory containing data files')
args = parser.parse_args()

if args.dir_name:
    dir_path = args.dir_name
    if dir_path[-1] != '/':
        dir_path += '/'
else:
    dir_path = easygui.diropenbox(msg = 'Please select data directory')

# Create plot dir
plot_dir = os.path.join(dir_path, "channel_profile_plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Get files to read
amp_files = glob.glob(os.path.join(dir_path, "amp*dat"))
amp_files = sorted(amp_files)
if len(amp_files) < 1:
    raise Exception("Couldn't find amp*.dat files in dir" + "\n" +\
            f"{dir_path}")
digin_files = sorted(glob.glob(os.path.join(dir_path, "board-DIN*")))

# Plot files
downsample = 100
row_lim = 8
row_num = np.min((row_lim, len(amp_files)))
col_num = int(np.ceil(len(amp_files)/row_num))


fig,ax = plt.subplots(row_num, col_num, 
        sharex=True, sharey=True, figsize = (15,10))
for this_file, this_ax in tqdm(zip(amp_files, ax.flatten())):
    data = np.fromfile(this_file, dtype = np.dtype('int16'))
    this_ax.plot(data[::downsample])
    this_ax.set_ylabel("_".join(os.path.basename(this_file)\
            .split('.')[0].split('-')[1:]))
plt.suptitle('Amplifier Data')
fig.savefig(os.path.join(plot_dir, 'amplifier_data'))
plt.close(fig)

fig,ax = plt.subplots(len(digin_files),
        sharex=True, sharey=True, figsize = (8,10))
for this_file, this_ax in tqdm(zip(digin_files, ax.flatten())):
    data = np.fromfile(this_file, dtype = np.dtype('uint16'))
    this_ax.plot(data[::downsample])
    this_ax.set_ylabel("_".join(os.path.basename(this_file)\
            .split('.')[0].split('-')[1:]))
plt.suptitle('DIGIN Data')
fig.savefig(os.path.join(plot_dir, 'digin_data'))
plt.close(fig)
