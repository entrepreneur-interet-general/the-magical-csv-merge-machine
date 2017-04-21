#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 19:48:22 2017

@author: leo
"""

import hashlib
import json
import os
import random
import time

def gen_proj_id():
    '''Generate unique non-guessable string for project ID'''
    unique_string = str(time.time()) + '_' + str(random.random())
    h = hashlib.md5()
    h.update(unique_string.encode('utf-8'))
    project_id = h.hexdigest()
    return project_id

NOT_IMPLEMENTED_MESSAGE = 'NOT IMPLEMENTED in abstract class'

class AbstractProject():
    def path_to(self, file_role='', module_name='', file_name=''):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)

    def create_metadata(self, description=''):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)       
        
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 description=''):
        
        if (project_id is None) and (not create_new):
            raise Exception('Set create_new to True or specify project_id')

        if create_new: 
            # Generate project id if none is passed
            if project_id is None:
                self.project_id = gen_proj_id()
            else:
                self.project_id = project_id
            
            path_to_proj = self.path_to()
            
            if os.path.isdir(path_to_proj):
                raise Exception('Project already exists. Choose a new path or \
                                delete the existing: {}'.format(path_to_proj))
            else:
                os.makedirs(path_to_proj)
            
            # Create metadata
            self.metadata = self.create_metadata(description=description)
            self.write_metadata()
        else:
            self.project_id = project_id
            self.metadata = self.read_metadata()
            
        # Initiate with no data in memory
        self.mem_data = None
        self.mem_data_info = {} # Information on data in memory
        self.log_buffer = [] # List of logs not yet written to metadata.json    
    
        
    def upload_config_data(self, config_dict, file_role, module_name, file_name):
        '''Will write config file'''
        
        if config_dict is None:
            return

        # Create directories
        dir_path = self.path_to(file_role, module_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)   
        
        # Write file
        file_path = self.path_to(file_role, module_name, file_name)
        with open(file_path, 'w') as w:
            json.dump(config_dict, w)

    def read_config_data(self, file_role, module_name, file_name):
        '''
        Reads json file and returns dictionary (empty dict if file is not found)
        '''
        file_path = self.path_to(file_role=file_role, module_name=module_name, 
                                 file_name=file_name)
        if os.path.isfile(file_path):
            config = json.loads(open(file_path).read())
        else: 
            config = {}
        return config            
    
    def read_metadata(self):
        '''Wrapper around read_config_data'''
        metadata = self.read_config_data('', '', file_name='metadata.json')
        assert metadata['project_id'] == self.project_id
        return metadata
    
    def write_metadata(self):
        path_to_metadata = self.path_to(file_name='metadata.json')
        json.dump(self.metadata, open(path_to_metadata, 'w'))
        
        
    def remove(self, file_role, module_name='', file_name=''):
        '''Removes a file from the project'''
        self.check_file_role(file_role)
        file_path = self.path_to(file_role, module_name, file_name)
        
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            raise Exception('{0} (in: {1}, as: {1}) could not be found in \
                            project'.format(file_name, module_name, file_role))
