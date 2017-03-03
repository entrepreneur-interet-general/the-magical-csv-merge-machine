#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:04:51 2017

@author: leo

metadata.json:
    
{
timestamp: 1487940990.422995, # Creation time
user_id: ???, # If we track users
use_internal_ref: True,
internal_ref_name: sirene,
source_names = ['source_1.csv']
source_log:[ # List of modules that were executed with what source files (from what module). was there an error
    {file_name: "source_id_1", module: "load", timestamp: 1487949990.422995, origin: "INIT", error:False},
    {file_name: "source_id_1", module: "missing_values", timestamp: 1487949990.422995, origin: "load", error:False}
    ]
ref_log:[]
project_id:347a7ba113a8cb3863b0c40246ec9098
]
}

IDEAS:
    - split file

DEV GUIDELINES:
    - Modules take as input (pd.DataFrame, dict_for_parameters)
    - Current state should be fully understandable from metadata
    - Each module shall take care of creating it's own directory

TODO:
    - File load
    - Module transform
    - Make generic api call for single module
    - delete 
    - Change log structure ?
    - Make function to fetch last file conditional to arguments.
"""

import gc
import hashlib
import json
import os
import random
import shutil
import time

import pandas as pd

# IMPORT MODULES
from infer_nan import infer_mvs, replace_mvs

from CONFIG import DATA_PATH

def path_to_ref(referential_name, module_name='', file_name=''):
    '''
    Return path to directory that stores specific information for stored
    canonical_data
    '''
    dir_path = os.path.join(DATA_PATH, 'referentials', referential_name, module_name, file_name)
    return os.path.abspath(dir_path)

def gen_proj_id():
    '''Generate unique non-guessable string for project ID'''
    unique_string = str(time.time()) + '_' + str(random.random())
    h = hashlib.md5()
    h.update(unique_string)
    project_id = h.hexdigest()
    return project_id

def check_file_role(file_role):
    if file_role not in ['ref', 'source']:
        raise Exception('"file_role" is either "source" or "ref"')

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
    return {'source_log': [], 'ref_log': []}


class Project():
    def __init__(self, project_id=None):
        if project_id is None:            
            self.new_project()
        else:
            self.project_id = project_id
            self.metadata = self.read_metadata()
            
        # No data in memory initially
        self.mem_data = None
        self.mem_data_info = None # Information on data in memory
        self.log_buffer = init_log_buffer() # List of logs not yet written to metadata.json

    def init_log(self, module_name):
        log = {'file_name': self.mem_data_info['file_name'], 
               'origin': self.mem_data_info['module'],
               'module': module_name, 
               'start_timestamp': time.time(), 
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False}
        return log
        
    def end_log(self, log, error=False):
        log['end_timestamp'] = time.time()
        log['error'] = False
        return log
    
    
    def time_since_created(self):
        return time.time() - float(self.metadata['timestamp'])
    
    
    # TODO: add shared to metadata
    def time_since_last_action(self):
        last_time = float(self.metadata['timestamp'])
        for file_role in ['source', 'ref']:
            log_name = file_role + '_log'
            if self.metadata[log_name]:
                last_time = max(last_time, float(self.metadata[log_name][-1]['end_timestamp']))
        return time.time() - last_time

    def check_mem_data(self):
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` at least once')        

    def create_metadata(self):
        metadata = dict()
        metadata['timestamp'] = time.time()
        metadata['user_id'] = 'NOT IMPlEMENTED'
        metadata['use_internal_ref'] = None
        metadata['internal_ref_name'] = None
        metadata['source_names'] = []
        metadata['source_log'] = []
        metadata['ref_log'] = []
        metadata['project_id'] = self.project_id
        return metadata

    def new_project(self):
        '''Create new project'''
        # Create directory
        self.project_id = gen_proj_id()
        path_to_proj = self.path_to()
        if not os.path.isdir(path_to_proj):
            os.makedirs(path_to_proj)
        # Create metadata
        self.metadata = self.create_metadata()
        self.write_metadata()   

    def delete_project(self):
        '''Deletes entire folder containing the project'''
        path_to_proj = self.path_to()
        shutil.rmtree(path_to_proj)
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        assert file_role in ['', 'source', 'ref']
        #assert all((not x[-1]) or all(x[:-1]) for i in [[project_id, file_role, module_name, file_name][:i] for i in range(4)]) 
        dir_path = os.path.join(DATA_PATH, 'projects', self.project_id, file_role, module_name, file_name)
        return os.path.abspath(dir_path)
    
    def add_init_data(self, file, file_role, file_name):
        """
        Add source or reference to the project. Will write. Can only add table 
        as INIT (not in modules) by design.
        """
        check_file_role(file_role)
        
        #
        self.mem_data_info = {'file_role': file_role, 
                               'file_name': file_name,
                               'module': 'INIT'}
        log = self.init_log('INIT')
        
        # TODO: add separator detection
        self.mem_data = pd.read_csv(file, encoding=None, dtype='unicode')

        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer[self.mem_data_info['file_role'] + '_log'].append(log)
        
        self.write_data()
        self.clear_memory()
    
    def remove_data(self, file_role, module_name='', file_name=''):
        '''Removes the corresponding data file'''
        check_file_role(file_role)
        file_path = self.path_to(file_role, module_name, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            raise Exception('{0} (in: {1}, as: {1}) could not be found in project'.format(file_name, module_name, file_role))
    
    def read_metadata(self):
        path_to_metadata = self.path_to(file_name='metadata.json')
        metadata = json.loads(open(path_to_metadata).read())
        assert metadata['project_id'] == self.project_id
        return metadata
    
    def write_metadata(self):
        path_to_metadata = self.path_to(file_name='metadata.json')
        json.dump(self.metadata, open(path_to_metadata, 'w'))
    
    def clean_metadata(self, file_role, file_name):
        '''Remove all mentions of a file in metadata'''
        check_file_role(file_role)
        log_name = file_role + '_log'
        self.metadata[log_name] = filter(lambda x: x['file_name']!=file_name, self.metadata[log_name])
    
    def get_arb(self):
        '''List directories and files in project'''
        path_to_proj = self.path_to()
        return get_arb(path_to_proj)
        
    
    def get_last_successful_written_module(self, file_role, file_name):
        '''
        Get name of the last module that was run with the given source and file 
        name.
        
        INPUT:
            - ...

        OUTPUT:
            - module_name ('INIT' if no previous module)
        '''
        check_file_role(file_role)

        # Check that original source file exists (TODO: should check in log instead ?)
        if not os.path.isfile(self.path_to(file_role=file_role, module_name='INIT', 
                                           file_name=file_name)):
            raise Exception('{0} (as: {1}) could not be found in INIT folder \
                            (not uploaded?)'.format(file_name, file_role))
            
        log_name = file_role + '_log'
        
        module_name = 'INIT'
        for log in self.metadata[log_name][::-1]:
            if not log['error'] and (log['file_name'] == file_name):
                module_name = log['module']
                break
        return module_name
        
    #    def get_last_written(self, file_role=None, module_name=None, file_name=None, error=False):
    #
    #        if 
    #        for log in self.metadata[log_name][::-1]:
    #            if not log['error'] and (log['file_name'] == file_name):
    #                module_name = log['module']
    #                break        
    
    def load_data(self, file_role, module_name, file_name):
        '''Load data as pandas DataFrame'''
        file_path = self.path_to(file_role, module_name, file_name)
        self.mem_data = pd.read_csv(file_path, encoding='utf-8')
        self.mem_data_info = {'file_role': file_role, 
                               'file_name': file_name,
                               'module': module_name}
    
    def write_log_buffer(self, written):
        '''
        Appends log buffer to metadata, writes metadata and clears log_buffer.
        
        INPUT: 
            - written: weather or not the output was written somewhere
        
        '''
        # Add log buffer to metadata
        for key in ['source_log', 'ref_log']:
            if self.log_buffer[key]:
                self.log_buffer[key][-1]['written'] = True
                self.metadata[key].extend(self.log_buffer[key])

        # Write metadata and clear log buffer
        self.write_metadata()
        self.log_buffer = init_log_buffer()

        
    def write_data(self):
        '''Write data (and metadata) in memory to proper module'''
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

        
    def clear_memory(self):
        '''Removes the table loaded in memory'''
        self.mem_data = None
        self.mem_data_info = None
        gc.collect()
    
    def transform(self, module_name, params):
        '''Run module on pandas DataFrame in memory and update memory state'''
        
        MODULES = {'replace_mvs': replace_mvs
                    }

        self.check_mem_data()        
        
        # Initiate log
        log = self.init_log(module_name)

        # TODO: catch module errors and add to log
        # Run module on pandas DataFrame 
        self.mem_data = MODULES[module_name](self.mem_data, params)
        self.mem_data_info['module'] = module_name
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer[self.mem_data_info['file_role'] + '_log'].append(log)
 
    def infer(self, module_name, params):
        '''Just runs the module name and returns answer'''
        MODULES = {'infer_mvs': infer_mvs
                    }
       
        self.check_mem_data()  
        # Initiate log
        log = self.init_log(module_name)
            
        infered_params = MODULES[module_name](self.mem_data, params)
        
        # Update log buffer
        self.log_buffer[self.mem_data_info['file_role'] + '_log'].append(log)    
                
        # TODO: write result of inference
        return infered_params
    
    
if __name__ == '__main__':
    # Create/Load a project
    project_id = "347a7ba113a8cb3863b0c40246ec9098"
    proj = Project(project_id)
    
    # Upload source to project
    file_path = 'local_test_data/source.csv'
    with open(file_path) as f:
        proj.add_init_data(f, 'source', 'source.csv')
        
    # Load source data to memory
    proj.load_data(file_role='source', module_name='INIT' , file_name='source.csv')
    
    # Try transformation
    params = {'mvs_dict': {'all': [], 'columns': {u'uai': [(u'NR', 0.2, ['len_ratio'])]}}, 
            'thresh': 0.6}
    proj.transform('replace_mvs', params)
    
    # Write transformed file
    proj.write_data()
    
    # Remove previously uploaded file
    # proj.remove_data('source', 'INIT', 'source.csv')    
    import pprint
    pprint.pprint(proj.get_arb())
    pprint.pprint(proj.metadata)
    
    proj.clean_metadata('source', 'source.csv') 