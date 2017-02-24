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
    - 

TODO:
    - File load
    - Module transform
    - Make generic api call for single module
    - delete 
"""

import hashlib
import json
import os
import random
import time

DATA_PATH = 'data'

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
    

class Project():
    def __init__(self, project_id=None):
        if project_id is None:            
            self.new_project()
        else:
            self.project_id = project_id
            self.metadata = self.read_metadata()

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
        os.remove(path_to_proj)
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        #assert all((not x[-1]) or all(x[:-1]) for i in [[project_id, file_role, module_name, file_name][:i] for i in range(4)]) 
        dir_path = os.path.join(DATA_PATH, 'projects', self.project_id, file_role, module_name, file_name)
        return os.path.abspath(dir_path)
    
    def add_table(self, file, file_role, file_name):
        """
        Add source or reference to the project. Will write. Can only add table 
        as INIT (not in modules) by design.
        """
        check_file_role(file_role)
        
        file_path = self.path_to(file_role, file_name=file_name)
        
        if os.path.isfile(file_path):
            raise Exception('{0} (as: {1}) already exists'.format(file_name, file_role))
    
    def remove_table(self, file_role, module_name='', file_name=''):
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
    
    def get_last_successful_module_name(self, file_role, file_name):
        '''
        Get name of the last module that was run with the given source and file 
        name.
        
        OUTPUT:
            - module_name ('INIT' if no previous module)
        '''
        check_file_role(file_role)

        # Check that original source file exists (TODO: should check in log instead ?)
        if not os.path.isfile(self.path_to(file_role=file_role, file_name=file_name)):
            raise Exception('{0} (as: {1}) could not be found in project'.format(file_name, file_role))
            
        log_name = file_role + '_log'
        
        module_name = 'INIT'
        for log in self.metadata[log_name][::-1]:
            if not log['error'] and (log['file_name'] == file_name):
                module_name = log['module']
                break
        return module_name
            
    def run_on_single_file(self, file_role, module_name, file_name):
        '''Run module on single file'''
        print ''
        
        
        
        
        # TODO: update metadata and write
        
    
if __name__ == '__main__':
    # Try creating a project
    
    project_id = "347a7ba113a8cb3863b0c40246ec9098"
    proj = Project(project_id)