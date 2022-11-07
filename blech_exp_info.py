"""
For help with input arguments:
    python blech_exp_info.py -h 


Code to generate file containing relevant experimental info:

X Animal name
X Exp Type
X Date
X Time Stamp
X Regions Recorded from
X Electrode Layout According to Regions
X Taste concentrations and dig_in order
X Taste Palatability Ranks
X Laser parameters and dig_in
X Misc Notes
"""

import glob
import json
import numpy as np
import os
import easygui
import sys
import re
import itertools as it
import argparse
import pdb
import pandas as pd
from utils.blech_utils import entry_checker


# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(description = 'Creates files with experiment info')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('--template', '-t', 
        help = 'Template (.info) file to copy experimental details from')
parser.add_argument('--mode', '-m', default = 'legacy', 
                    choices = ['legacy','updated'])
args = parser.parse_args()

if args.dir_name:
    dir_path = args.dir_name
    if dir_path[-1] != '/':
        dir_path += '/'
else:
    dir_path = easygui.diropenbox(msg = 'Please select data directory')

dir_name = os.path.basename(dir_path[:-1])

# Extract details from name of folder
splits = dir_name.split("_")
this_dict = {
        "name" : splits[0],
        "exp_type" : splits[1],
        "date": splits[2],
        "timestamp" : splits[3]}

##################################################
## Brain Regions and Electrode Layout
##################################################

if args.template:
    with open(args.template,'r') as file:
        template_dict = json.load(file)
        var_names = ['regions','ports','electrode_layout','taste_params',
                'laser_params','notes']

        from_template = {key:template_dict[key] for key in var_names}
        fin_dict = {**this_dict,**from_template}

else:

    def word_check(x):
        words = re.findall('[A-Za-z]+',x)
        return (sum([i.isalpha() for i in words]) == len(words)) and len(words) > 0

    region_str, continue_bool = entry_checker(\
            msg = ' Which regions were recorded from (anything separated)  :: ',
            check_func = word_check,
            fail_response = 'Please enter letters only')
    if continue_bool:
        regions = [x.lower() for x in re.findall('[A-Za-z]+',region_str)]
    else:
        exit()

    # Find all ports used
    file_list = os.listdir(dir_path)
    ports = list(set(f[4] for f in file_list if f[:3] == 'amp'))
    # Sort the ports in alphabetical order
    ports.sort()

    # Write out file and ask user to define regions in file
    layout_file_path = os.path.join(dir_path, dir_name + "_electrode_layout.csv")

    def yn_check(x):
        return x in ['y','yes','n','no']

    if os.path.exists(layout_file_path):

        use_csv_str, continue_bool = entry_checker(\
                msg = "Layout file detected...use what's there? (y/yes/no/n)",
                check_func = yn_check,
                fail_response = 'Please [y, yes, n, no]')
    else:
        use_csv_str = 'n'

    if use_csv_str in ['n','no']: 
        electrode_files = sorted([x for x in file_list if 'amp' in x])
        port_list = [x.split('-')[1] for x in electrode_files]
        electrode_num_list = [x.split('-')[2].split('.')[0] for x in electrode_files]
        layout_frame = pd.DataFrame()
        layout_frame['filename'] = electrode_files
        layout_frame['port'] = port_list
        layout_frame['electrode_num'] = electrode_num_list
        layout_frame['electrode_ind'] = layout_frame.index
        layout_frame['CAR_group'] = pd.Series()

        layout_frame = \
                layout_frame[['filename','electrode_ind',
                    'electrode_num','port','CAR_group']]

        layout_frame.to_csv(layout_file_path, index=False) 

        acceptable_cars = regions.copy()
        acceptable_cars.append('emg')
        prompt_str = 'Please fill in car groups / regions' + "\n" + \
                f"Acceptable inputs are {acceptable_cars}" + "\n" +\
                "Indicate different CARS from same region as GC1,GC2...etc"
        print(prompt_str)

        def confirm_check(x):
            this_bool = x in ['y','yes']
            return this_bool 
        perm_str, continue_bool = entry_checker(\
                msg = f'Lemme know when its done (y/yes) ::: ',
                check_func = confirm_check,
                fail_response = 'Please say y or yes')
        if not continue_bool:
            print('Welp...')
            exit()

    layout_frame_filled = pd.read_csv(layout_file_path)
    layout_frame_filled['CAR_group'] = layout_frame_filled['CAR_group'].str.lower() 
    layout_frame_filled['CAR_group'] = [x.strip() for x in \
            layout_frame_filled['CAR_group'] ]
    layout_dict = dict(list(layout_frame_filled.groupby('CAR_group').electrode_ind))
    for key,vals in layout_dict.items():
        layout_dict[key] = [layout_dict[key].to_list()]

    if 'emg' in layout_dict.keys():
        #orig_emg_electrodes = layout_dict.pop('emg').values()
        orig_emg_electrodes = layout_dict.pop('emg')[0]
        fin_emg_port = layout_frame_filled.port.loc[
                        layout_frame_filled.electrode_ind.isin(orig_emg_electrodes)].\
                        unique()
        fin_emg_port = list(fin_emg_port)
    else:
        fin_emg_port = []
        orig_emg_electrodes = []

    fin_perm = layout_dict

    ##################################################
    ## Dig-Ins
    ##################################################
    def count_check(x):
        nums = re.findall('[0-9]+',x)
        return sum([x.isdigit() for x in nums]) == len(nums)

    taste_dig_in_str, continue_bool = entry_checker(\
            msg = ' Taste dig_ins used (IN ORDER, anything separated)  :: ',
            check_func = count_check,
            fail_response = 'Please enter integers only')
    if continue_bool:
        nums = re.findall('[0-9]+',taste_dig_in_str)
        taste_digins = [int(x) for x in nums]
    else:
        exit()
    
    def float_check(x):
        global taste_digins
        return len(x.split(',')) == len(taste_digins)

    # Trials per taste
    dig_in_trials_str, continue_bool = entry_checker(\
            msg = ' Trials per dig-in (IN ORDER, anything separated)  :: ',
            check_func = count_check,
            fail_response = 'Please enter integers only, and as many as dig-ins')
    if continue_bool:
        nums = re.findall('[0-9]+',dig_in_trials_str)
        dig_in_trials = [int(x) for x in nums]
    else:
        exit()

    def taste_check(x):
        global taste_digins
        return len(re.findall('[A-Za-z]+',x)) == len(taste_digins)
    taste_str, continue_bool = entry_checker(\
            msg = ' Tastes names used (IN ORDER, anything separated)  :: ',
            check_func = taste_check,
            fail_response = 'Please enter as many tastes as digins')
    if continue_bool:
        tastes = re.findall('[A-Za-z]+', taste_str)
    else:
        exit()

    conc_str, continue_bool = entry_checker(\
            msg = 'Corresponding concs used (in M, IN ORDER, COMMA separated)  :: ',
            check_func = float_check,
            fail_response = 'Please enter as many concentrations as digins')
    if continue_bool:
        concs = [float(x) for x in conc_str.split(",")]
    else:
        exit()

    # Ask user for palatability rankings
    def pal_check(x):
        global taste_digins
        nums = re.findall('[1-9]+',x)
        return  sum([x.isdigit() for x in nums]) == len(nums) and \
                sum([1<=int(x)<=len(taste_digins) for x in nums]) == len(taste_digins)

    taste_fin = str(list(zip(taste_digins, list(zip(tastes,concs)))))
    palatability_str, continue_bool = entry_checker(\
            msg = f' {taste_fin} \n Enter palatability rankings used (anything separated), higher number = more palatable  :: ',
            check_func = pal_check,
            fail_response = 'Please enter numbers 1<=x<len(tastes)')
    if continue_bool:
        nums = re.findall('[1-9]+',palatability_str)
        pal_ranks = [int(x) for x in nums]
    else:
        exit()

    ########################################
    # Ask for laser info
    laser_select_str, continue_bool = entry_checker(\
            msg = 'Laser dig_in , <BLANK> for none::: ',
            check_func = count_check,
            fail_response = 'Please enter a single, valid integer')
    if continue_bool:
        if len(laser_select_str) == 0:
            laser_digin = []
        else:
            laser_digin = [int(laser_select_str)]
    else:
        exit()

    def laser_check(x):
        nums = re.findall('[0-9]+',x)
        return sum([x.isdigit() for x in nums]) == 2 
    if laser_digin:
        laser_select_str, continue_bool = entry_checker(\
                msg = 'Laser onset_time, duration (ms, IN ORDER, anything separated) ::: ',
                check_func = laser_check,
                fail_response = 'Please enter two, valid integers')
        if continue_bool:
            nums = re.findall('[0-9]+',laser_select_str)
            onset_time, duration = [int(x) for x in nums]
        else:
            exit()
    else:
        onset_time, duration = [None, None]

    notes = input('::: Please enter any notes about the experiment. \n ::: ')

    ########################################
    ## Finalize dictionary
    ########################################

    fin_dict = {**this_dict,
            'regions' : regions,
            'ports' : ports,
            'emg' : {\
                    'port': fin_emg_port, 
                    'electrodes' : orig_emg_electrodes},
            'electrode_layout' : fin_perm,
            'taste_params' : {\
                    'dig_ins' : taste_digins,
                    'trial_count' : dig_in_trials,
                    'tastes' : tastes,
                    'concs' : concs,
                    'pal_rankings' : pal_ranks},
            'laser_params' : {\
                    'dig_in' : laser_digin,
                    'onset' : onset_time,
                    'duration': duration},
            'notes' : notes}


json_file_name = os.path.join(dir_path,'.'.join([dir_name,'info']))
with open(json_file_name,'w') as file:
    json.dump(fin_dict, file, indent = 4)

