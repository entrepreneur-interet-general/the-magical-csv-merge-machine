#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:04:51 2017

@author: leo

Abstract class that deals with csv data, transformations, and inference.


IDEAS:
    - split file

DEV GUIDELINES:
    - Modules take as input (pd.DataFrame, dict_for_parameters)
    - Current state should be fully understandable from metadata
    - Each module shall take care of creating it's own directory
    - File name is unique (not accross reference)

TODO:
    - Store user params
    - Deal with log for merge 
"""

import gc
import hashlib
import itertools
import json
import os
import random
import shutil
import time

import pandas as pd

from infer_nan import infer_mvs, replace_mvs
from dedupe_linker import dedupe_linker

MODULES = {
        'transform':{'replace_mvs': replace_mvs},
        'infer':{'infer_mvs': infer_mvs},
        'link': {'dedupe_linker': dedupe_linker}
        }


def gen_proj_id():
    '''Generate unique non-guessable string for project ID'''
    unique_string = str(time.time()) + '_' + str(random.random())
    h = hashlib.md5()
    h.update(unique_string.encode('utf-8'))
    project_id = h.hexdigest()
    return project_id

def allowed_file(filename):
    '''Check if file name is correct'''
    # Check if extension is .csv
    return filename[-4:] == '.csv'
    
def get_arb(dir_path):
    '''Return arborescence as dict'''
    to_return = dict()
    for name in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, name)):
            to_return[name] = get_arb(os.path.join(dir_path, name))
        else:
            to_return[name] = {}
    return to_return



def init_log_buffer():
    return []


NOT_IMPLEMENTED_MESSAGE = 'NOT IMPLEMENTED in abstract class'

class Project():
    """
    Abstract class to deal with data, data transformation, and metadata.
    
    SUMMARY:
        This class allows to load user data and perform inference or 
        transformations. Before and after transformation, data is stored in 
        memory as Pandas DataFrame. Transformations are only written to disk if
        write_data is called. A log that describes the changes made to the 
        data in memory is stored in log_buffer. When writing data, you should
        also write the log_buffer (write_log_buffer) to log the changes that 
        were performed.
    
    In short: Objects stored in memory are:
        - TODO: write this
    
    """

#==============================================================================
# Methods that should be implemented in children
#==============================================================================

    def check_file_role(self, file_role):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)

    def path_to(self, file_role='', module_name='', file_name=''):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)

    def create_metadata(self):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)    
        
#==============================================================================
# Actual class
#==============================================================================

    def __init__(self, project_id=None, create_new=False):
        if create_new: 
            self.new_project()
        else:
            if project_id is None:
                raise Exception('Set create_new to true or specify project_id')
            self.project_id = project_id
            self.metadata = self.read_metadata()
            
        # No data in memory initially
        self.mem_data = None
        self.mem_data_info = {} # Information on data in memory
        self.log_buffer = init_log_buffer() # List of logs not yet written to metadata.json    
    
    def new_project(self, project_id=None):
        '''
        Create new project. Will use a specified project_id or generate it 
        otherwise
        '''
        # Create directory
        if project_id is None:
            self.project_id = gen_proj_id()
        else:
            self.project_id = project_id
        
        path_to_proj = self.path_to()
        
        if os.path.isdir(path_to_proj):
            raise Exception('Project already exists. \
                    Choose a new path or delete the existing: {}'.format(path_to_proj))
        else:
            os.makedirs(path_to_proj)
        # Create metadata
        self.metadata = self.create_metadata()
        self.write_metadata()
    
    def init_log(self, module_name, module_type):
        assert module_type in ['transform', 'infer', 'link']
        log = { # Data being modified
               'file_name': self.mem_data_info.get('file_name', None), 
               'origin': self.mem_data_info.get('module', None),
               'file_role': self.mem_data_info.get('file_role', None),
                # Modification at hand                        
               'module': module_name, # Module to be executed
               'module_type': module_type, # Type (transform, infer, or dedupe)
               'start_timestamp': time.time(),
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False}
        return log
        
    def end_log(self, log, error=False):
        log['end_timestamp'] = time.time()
        log['error'] = error
        return log
    
    def time_since_created(self):
        return time.time() - float(self.metadata['timestamp'])
    
    # TODO: add shared to metadata
    def time_since_last_action(self):
        last_time = float(self.metadata['timestamp'])
        if self.metadata['log']:
            last_time = max(last_time, float(self.metadata['log'][-1]['end_timestamp']))
        return time.time() - last_time

    def get_arb(self):
        '''List directories and files in project'''
        path_to_proj = self.path_to()
        return get_arb(path_to_proj)


    def list_files(self, extensions=['.csv']):
        '''
        Lists csv files (from data) in data directory and presents a list of modules in 
        which they are present. You can combine this with get_last_written
        '''
        all_files = dict()
        for file_role in ['ref', 'source', 'link']:
            all_files[file_role] = dict()
            root_path = self.path_to(file_role=file_role)
            if os.path.isdir(root_path):
                for _dir in os.listdir(root_path):
                    if os.path.isdir(os.path.join(root_path, _dir)):
                        for file_name in os.listdir(os.path.join(root_path, _dir)):
                            if any(file_name[-len(ext):] == ext for ext in extensions):
                                if file_name not in all_files[file_role]:
                                    all_files[file_role][file_name] = [_dir]
                                else:
                                    all_files[file_role][file_name].append(_dir)
        return all_files

    def log_by_file_name(self):
        # Sort by name and date        
        sorted_log = sorted(self.metadata['log'], key=lambda x: (x['file_name'], 
                                x['start_timestamp']))
        
        resp = dict()
        for key, group in itertools.groupby(sorted_log, lambda x: x['file_name']):
            resp[key] = list(group)
        return resp
            


    def get_last_written(self, file_role=None, module=None, file_name=None):
        '''
        Return info on data that was last successfully written (from log)
        
        INPUT:
            - file_role: filter on file role (last data with given file_role)
            - module: filter on given module
            - file_name: filter on file_name
            
        OUTPUT:
            - module_name ('INIT' if no previous module)
        '''
        self.check_file_role(file_role)
    
        for log in self.metadata['log'][::-1]:
            if (not log['error']) and log['written'] \
                      and ((log['file_role'] == file_role) or file_role is None) \
                      and ((log['module'] == module) or module is None) \
                      and ((log['file_name'] == file_name) or file_name is None):
                break
        else:
            raise Exception('No written data could be found in logs')
        file_role = log['file_role']
        module = log['module']
        file_name = log['file_name']        
        return (file_role, module, file_name)

    def check_mem_data(self):
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')        

    def delete_project(self):
        '''Deletes entire folder containing the project'''
        path_to_proj = self.path_to()
        shutil.rmtree(path_to_proj)
    
    def add_init_data(self, file, file_role, file_name):
        """
        Add source or reference to the project. Will write. Can only add table 
        as INIT (not in modules) by design.
        """
        self.check_file_role(file_role)
        
        # Check that file is not already present
        
        #
        self.mem_data_info = {'file_role': file_role, 
                              'file_name': file_name,
                              'module': 'INIT'}
        log = self.init_log('INIT', 'transform')
        
        # TODO: add separator detection
        self.mem_data = pd.read_csv(file, encoding=None, dtype='unicode')

        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        
        self.write_data()
        self.write_log_buffer(written=True)
        self.clear_memory()
    
    def add_config_data(self, config_dict, file_role, module, file_name):
        '''Will write config file'''
        if (file_role != 'link') or (module != 'dedupe_linker'):
            raise Exception('Can only upload config files to file_role: link\
                            and module:dedupe_linker') # TODO: Is this good? Yes/No?
            
        if file_name not in ['training.json', 'column_matches.json']:
            raise Exception('For now you can only upload files named training.json or column_matches.json')

        # Create directories
        dir_path = self.path_to(file_role, module)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)   
        
        # Write file
        file_path = self.path_to(file_role, module, file_name)
        with open(file_path, 'w') as w:
            json.dump(config_dict, w)
            
    

    
    def remove_data(self, file_role, module_name='', file_name=''):
        '''Removes the corresponding data file'''
        self.check_file_role(file_role)
        file_path = self.path_to(file_role, module_name, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            raise Exception('{0} (in: {1}, as: {1}) could not be found in project'.format(file_name, module_name, file_role))
 
    def read_config_data(self, file_role, module, file_name):
        '''Reads json file'''
        file_path = self.path_to(file_role=file_role, module_name=module, file_name=file_name)
        if os.path.isfile(file_path):
            config = json.loads(open(file_path).read())
        else: 
            config=None
        return config    
    
    def read_metadata(self):
        '''Wrapper around read_config_data'''
        metadata = self.read_config_data('', '', file_name='metadata.json')
        assert metadata['project_id'] == self.project_id
        return metadata
    
    def write_metadata(self):
        path_to_metadata = self.path_to(file_name='metadata.json')
        json.dump(self.metadata, open(path_to_metadata, 'w'))

    def remove(self, file_name):
        '''Remove all files and clear metadata with given name from project'''
        # TODO: deal with .csv dependency
        all_files = self.list_files(extensions=['.csv'])
        
        for file_role, name_dict in all_files.items():
            for _file_name, module_name in name_dict.items():
                if file_name == _file_name:
                    self.remove_data(file_role, module_name, file_name)
        
        self.metadata['log'] = filter(lambda x: (x['file_name']!=file_name), 
                                                 self.metadata['log'])     
        self.write_metadata()
        
        
    def load_data(self, file_role, module_name, file_name):
        '''Load data as pandas DataFrame'''
        file_path = self.path_to(file_role, module_name, file_name)
        self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype='unicode')
        self.mem_data_info = {'file_role': file_role, 
                               'file_name': file_name,
                               'module': module_name}
        
    def get_sample(self, file_role, module_name, file_name, row_idxs=range(5)):
        '''Returns a dict with the selected rows (including header)'''
        file_path = self.path_to(file_role, module_name, file_name)
        if row_idxs is not None:
            assert row_idxs[0] == 0
            max_rows = max(row_idxs)    
            tab = pd.read_csv(file_path, encoding='utf-8', dtype='unicode', nrows=max_rows)
            tab = tab.iloc[[x - 1 for x in row_idxs], :]
        else:
            tab = pd.read_csv(file_path, encoding='utf-8', dtype='unicode')
        return tab.to_dict('records')
        
    
    def write_log_buffer(self, written):
        '''
        Appends log buffer to metadata, writes metadata and clears log_buffer.
        
        INPUT: 
            - written: weather or not the data was written
        
        '''
        if not self.log_buffer:
            raise Exception('No log buffer or write; no operations since last write')
        
        # Indicate if any data was written
        if written:
            for log in self.log_buffer[::-1]:
                assert log['module_type'] in ['infer', 'transform', 'link']
                if log['module_type'] in ['transform', 'link']:
                    log['written'] = True
                    break
                       
        # Add buffer to metadata  
        self.metadata['log'].extend(self.log_buffer)
    
        # Write metadata and clear log buffer
        self.write_metadata()
        self.log_buffer = init_log_buffer()


    def write_data(self):
        '''Write data stored in memory to proper module'''
        self.check_mem_data()
            
        # Write data
        dir_path = self.path_to(self.mem_data_info['file_role'], 
                                self.mem_data_info['module'])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(self.mem_data_info['file_role'], 
                                 self.mem_data_info['module'], 
                                 self.mem_data_info['file_name'])
        self.mem_data.to_csv(file_path, encoding='utf-8', index=False)


    def write_infered_params(self, params, file_role, module_name):
        '''
        Writes dictionary params in a file named "infered_params" at the 
        specified path.    
        '''
        dir_path = self.path_to(self.mem_data_info['file_role'], module_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(file_role, 
                                 module_name, 
                                 'infered_params.json')
        with open(file_path, 'w') as w:
            json.dump(params, w)

        
    def clear_memory(self):
        '''Removes the table loaded in memory'''
        self.mem_data = None
        self.mem_data_info = None
        gc.collect()


    def transform(self, module_name, params):
        '''Run module on pandas DataFrame in memory and update memory state'''
        self.check_mem_data()        
        
        # Initiate log
        log = self.init_log(module_name, 'transform')

        # TODO: catch module errors and add to log
        # Run module on pandas DataFrame 
        self.mem_data = MODULES['transform'][module_name](self.mem_data, params)
        self.mem_data_info['module'] = module_name
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        return log
    

    def infer(self, module_name, params):
        '''Just runs the module on pandas DataFrame data in memory and returns answer'''
        self.check_mem_data()  
        
        # Initiate log
        log = self.init_log(module_name, 'infer')
            
        infered_params = MODULES['infer'][module_name](self.mem_data, params)
                
        # Write result of inference
        self.write_infered_params(params, self.mem_data_info['file_role'], module_name)

        # Update log buffer
        self.log_buffer.append(log)     
        
        return infered_params
    
    

    

