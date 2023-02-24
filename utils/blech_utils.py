"""
Utilities to support blech_clust processing
"""
import easygui
import os
import glob
import json

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


class imp_metadata():
    def __init__(self, args):
        self.dir_name = self.get_dir_name(args)
        self.get_file_list()
        self.get_hdf5_name()
        self.get_params_path()
        self.get_info_path()
        self.load_params()
        self.load_info()

    def get_dir_name(self, args):
        if len(args) > 1:
            dir_name = os.path.abspath(args)
            if dir_name[-1] != '/':
                dir_name += '/'
        else:
            dir_name = easygui.diropenbox(msg = 'Please select data directory')
        return dir_name

    def get_file_list(self,):
        self.file_list = os.listdir(self.dir_name)
        
    def get_hdf5_name(self,):
        hdf5_name = ''
        for files in self.file_list:
            if files[-2:] == 'h5':
                hdf5_name = files
        self.hdf5_name = hdf5_name

    def get_params_path(self,):
        self.params_file_path = glob.glob(os.path.join(self.dir_name,'**.params'))[0]

    def load_params(self,):
        with open(self.params_file_path, 'r') as params_file_connect:
            self.params_dict = json.load(params_file_connect)

    def get_info_path(self,):
        self.info_file_path = glob.glob(os.path.join(self.dir_name, '**.info'))[0]

    def load_info(self,):
        with open(self.info_file_path, 'r') as info_file_connect:
            self.info_dict = json.load(info_file_connect)
