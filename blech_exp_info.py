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


# Get name of directory with the data files
# Create argument parser
parser = argparse.ArgumentParser(description = 'Creates files with experiment info')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('--template', '-t', 
        help = 'Template (.info) file to copy experimental details from')
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

    # Depending on number of regions and ports, list all permutations
    # of electrode and ask user to select from them

    type_16_16_vals = [list(map(int,np.arange(8)))+list(map(int,np.arange(24,32))),
                    list(map(int, np.arange(8,24)))]
    type_16_16_keys = ['0-7,24-31','8-23']
    type_16_16 = dict(zip(type_16_16_keys,type_16_16_vals))
    type_32 = {'0-31' : list(map(int,np.arange(32)))}
    port_vals = [np.arange(32)+(32*num) for num in range(len(ports))]
    ports_dict = dict(zip(ports,port_vals))

    if len(ports) == 1:
        if len(regions) > 1:
            region_permutations = list(it.permutations(regions))
            split_permutations = [list(zip(perm,type_16_16.keys())) \
                    for perm in region_permutations]
            permutations = split_permutations 
        else:
            permutations = list(zip(regions,type_32.keys())) 

    else: # Assuming there would be no splitting if you have multiple ports
        if len(regions) == 1:
            permutations = [tuple((zip(regions,[ports])))]
        elif len(regions) ==2:
            region_permutations = list(it.permutations(regions))
            permutations = []
            for this_region_perm in region_permutations:
                for count in np.arange(1,len(ports)):
                    permutations.append(\
                            [(this_region_perm[0],ports[:count]),
                                (this_region_perm[1],ports[count:])])

    # Ask user to select appropriate permutation
    perm_msg = "\n".join([str(x) for x in (list(zip(range(len(permutations)),permutations)))])
    def select_check(x):
        global permutations
        return x.isdigit() and len(re.findall('[0-9]+',x))==1 and 0<=int(x)<=len(permutations)

    perm_str, continue_bool = entry_checker(\
            msg = f'{perm_msg} \n Please select the correct layout ::: ',
            check_func = select_check,
            fail_response = 'Please enter a single, valid integer')
    if continue_bool:
        select_ind = int(perm_str)
        selected_perm = permutations[select_ind]
        fin_perm = []
        if len(ports) == 1:
            if len(regions) == 1:
                fin_perm.append([selected_perm[0], [type_32[selected_perm[1]]]])
            else:
                for this_region in selected_perm:
                    fin_perm.append([this_region[0], type_16_16[this_region[1]]])
        else:
            fin_perm = []
            for this_region in selected_perm:
                fin_perm.append([this_region[0],
                            [ports_dict[x].tolist() for x in this_region[1]]])
    else:
        exit()

    # Ask user for EMG electrodes and add them as a "region" to the fin_perm
    def count_check(x):
        nums = re.findall('[0-9]+',x)
        return sum([x.isdigit() for x in nums]) == len(nums)
    if len(ports) > 1:
        port_vals = list(zip(np.arange(len(ports)), ports))
        port_str = "\n".join([str(x) for x in port_vals])
        print(port_str)
        emg_port_str, continue_bool = entry_checker(\
                msg = 'EMG Port, <BLANK> for none::: ',
                check_func = count_check,
                fail_response = 'Please enter a single, valid integer')
        if continue_bool:
            if len(emg_port_str) == 0:
                emg_port = []
            else:
                emg_port = [int(emg_port_str)]
        else:
            exit()
    else:
        emg_port = [0]

    if len(emg_port) > 0:
        potential_electrodes = np.arange(32) + 32*emg_port
        print(f"Port : {ports[emg_port[0]]}, \n Electrodes : {potential_electrodes}")
        emg_elec_str, continue_bool = entry_checker(\
                msg = 'EMG Electrodes, <BLANK> for none (ANYTHING separated) ::: ',
                check_func = count_check,
                fail_response = 'Please enter integers')
        if continue_bool:
            if len(emg_elec_str) == 0:
                emg_electrodes = []
            else:
                emg_electrodes = [int(x) for x in re.findall('[0-9]+',emg_elec_str)]
        else:
            exit()

    # Walk through fin_perm and delete emg_electrodes where you find them 
    # and add them as a new region
    if 'emg_electrodes' in dir():
        for region in fin_perm:
            for group in region[1]:
                for elec in emg_electrodes:
                    ind = np.where(np.array(group) == elec)[0]
                    if len(ind) > 0:
                        del group[ind[0]]

        fin_perm.append(['emg',[emg_electrodes]])


    ##################################################
    ## Dig-Ins
    ##################################################
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
            msg = f' {taste_fin} \n Enter palatability rankings used (anything separated)  :: ',
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
            laser_digin = int(laser_select_str)
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

