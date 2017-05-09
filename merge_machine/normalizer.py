#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 19:39:45 2017

@author: leo
"""

import csv
import gc
import itertools
import os
import shutil
import time

import pandas as pd

from abstract_data_project import AbstractDataProject
from CONFIG import NORMALIZE_DATA_PATH
from MODULES import MODULES, MODULE_ORDER

class Normalizer(AbstractDataProject):
    """
    Abstract class to deal with data, data transformation, and metadata.
    
    SUMMARY:
        This class allows to load user data and perform inference or 
        transformations. Before and after transformation, data is stored in 
        memory as Pandas DataFrame. Transformations are only written to disk if
        write_data is called. A log that describes the changes made to the 
        data in memory is stored in log_buffer. Agter writing data, you should
        also write the log_buffer (write_log_buffer) to log the changes that 
        were performed.
    
    In short: Objects stored in memory are:
        - TODO: write this
    """
    
#==============================================================================
# Actual class
#==============================================================================   

    def __init__(self, project_id=None, create_new=False, description=None, display_name=None):
        super().__init__(project_id, create_new, description, display_name=display_name)

    def create_metadata(self, description=None, display_name=None):
        metadata = super().create_metadata(description=description, 
                                            display_name=display_name)
        metadata['files'] = dict() # Contains single file metadata
        return metadata   

    def init_log(self, module_name, module_type):
        '''
        Initiate a log (before a module call). Use end_log to complete log message
        '''
        assert module_type in ['transform', 'infer']
        log = { 
                # Data being modified
               'file_name': self.mem_data_info.get('file_name'), 
               'origin': self.mem_data_info.get('module_name'),
               
                # Modification at hand                        
               'module_name': module_name, # Module to be executed
               'module_type': module_type, # Type (transform, infer, or dedupe)
               'start_timestamp': time.time(),
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False
               }
        return log
        
    
    def time_since_created(self):
        return time.time() - float(self.metadata['timestamp'])
    
    # TODO: add shared to metadata
    # TODO: change this in generic using get_last...
    def time_since_last_action(self):
        last_time = float(self.metadata['timestamp'])
        if self.metadata['log']:
            last_time = max(last_time, float(self.metadata['log'][-1]['end_timestamp']))
        return time.time() - last_time


    def _list_files(self, extensions=['.csv']):
        '''
        Lists csv files (from data) in data directory and presents a list of modules in 
        which they are present. You can combine this with get_last_written
        '''
        def is_dir(root_path, x):
            return os.path.isdir(os.path.join(root_path, x))
        
        all_files = dict()
        root_path = self.path_to()
        for _dir in filter(lambda x: is_dir(root_path, x), os.listdir(root_path)):
            for file_name in os.listdir(os.path.join(root_path, _dir)):
                if any(file_name[-len(ext):] == ext for ext in extensions):
                    if file_name not in all_files:
                        all_files[file_name] = [_dir]
                    else:
                        all_files[file_name].append(_dir)
        return all_files

    def log_by_file_name(self):
        # Sort by name and date        
        sorted_log = sorted(self.metadata['log'], key=lambda x: (x['file_name'], 
                                x['start_timestamp']))
        
        resp = dict()
        for key, group in itertools.groupby(sorted_log, lambda x: x['file_name']):
            resp[key] = list(group)
        return resp

    def get_last_written(self, module_name=None, file_name=None, 
                         before_module=None):
        '''
        Return info on data that was last successfully written (from log)
        
        INPUT:
            - module_name: filter on given module
            - file_name: filter on file_name
            - before_module: (string with module name) Looks for file that was 
                              written in a module previous to before_module 
                              (in the order defined by MODULE_ORDER)
            
        OUTPUT:
            - (module_name, file_name)
        '''        
        if (module_name is not None) and (before_module is not None):
            raise Exception('Variables module_name and before_module cannot be \
                            set simultaneously')
        
        previous_modules = {MODULE_ORDER[i]: MODULE_ORDER[:i] for i in range(len(MODULE_ORDER))}
    
        for log in self.metadata['log'][::-1]:
            if (not log['error']) and log['written'] \
                      and ((module_name is None) or (log['module_name'] == module_name)) \
                      and ((file_name is None) or (log['file_name'] == file_name)) \
                      and ((before_module is None) or (log['module_name'] in previous_modules[before_module])):                
                break
        else:
            raise Exception('No written data could be found in logs')
        
        module_name = log['module_name']
        file_name = log['file_name']        
        return (module_name, file_name)


    def path_to_last_written(self, module_name=None, file_name=None, before_module=None):
        (module_name, file_name) = self.get_last_written(module_name,
                                                        file_name, before_module)
        path = self.path_to(module_name, file_name)
        return path
        
    def safe_filename(self, file_name, ext='.csv'):
        assert file_name[-len(ext):] == ext        
        return "".join([c for c in file_name[:-4] if c.isalpha() or c.isdigit() or c==' ']).rstrip() + ext
    
    def upload_init_data(self, file, file_name, user_given_name=None):
        # TODO: deal with og_file_name, file_id, display_name, user_given_name
        
        """
        Upload and write source or reference to the project. Tables will
        be added to the "INIT" module.
        
        The file will be re-coded in utf-8 with a "," separator. Also, chars
        specified in CHARS_TO_REPLACE will be replaced by "_" in the header.
        """
        CHARS_TO_REPLACE = [' ', ',', '.', '(', ')', '\'', '\"', '/']
        ENCODINGS = ['utf-8', 'windows-1252']
        SEPARATORS = [',', ';', '\t']
        
        # Check that 
        
        # Check that user given name is not illegal
        if user_given_name is not None:
            if (len(user_given_name) < 4) or (user_given_name[-4:] != '.csv'):
                raise Exception('user given name should end with .csv')
            if any(x in user_given_name[:-4] for x in CHARS_TO_REPLACE):
                raise Exception('user_given_name sould be alphanumeric or underscores (+.csv)')
                
        # TODO: Check that file is not already present
        self.mem_data_info = {
                                'og_file_name': file_name,
                                'module_name': 'INIT'
                             }
    
        if user_given_name is None:
            self.mem_data_info['file_name'] = self.safe_filename(file_name)
            display_name = file_name
        else:
            file_name = user_given_name
            self.mem_data_info['file_name'] = file_name
            display_name = user_given_name
        
        # Check that file name is not already present 
        if file_name in self.metadata['files']:
            raise Exception('File: {0} already exists. Delete this file ' \
                             + 'or choose another name'.format(file_name))
        
        log = self.init_log('INIT', 'transform')
        could_read = False
        for encoding in ENCODINGS:
            for sep in SEPARATORS:
                try:
                    self.mem_data = pd.read_csv(file, sep=sep, encoding=encoding, dtype=str)
                    for char in CHARS_TO_REPLACE:
                        self.mem_data.columns = [x.replace(char, '_') for x in self.mem_data.columns]
                    could_read = True
                    break
                except Exception as e:
                    print(e)
                    file.seek(0)
            if could_read:
                break        
            
        if not could_read:
            raise Exception('Separator and/or Encoding not detected. Try uploading \
                            a csv with "," as separator with utf-8 encoding')

        # Add file to metadata
        self.metadata['files'][file_name] = {
                                                'og_file_name': file_name,
                                                'display_name': display_name,
                                                'upload_time': time.time()
                                            }
        self.write_metadata()
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        
        # write data and log
        self.write_data()
        self.write_log_buffer(written=True)

        # Write configuration (sep, encoding) to INIT dir
        config_dict = {
                        'sep': sep, 
                        'encoding': encoding, 
                        'nrows': self.mem_data.shape[0], 
                        'ncols': self.mem_data.shape[1]
                    }
        proj.upload_config_data(config_dict, 'INIT', 'infered_config.json')

        self.clear_memory()    

    def remove_all(self, file_name):
        '''
        Remove all occurences of files with a given file_name from the project
        and clears all mentions of this file_name in metadata
        '''
        # TODO: deal with .csv dependency
        all_files = self._list_files(extensions=['.csv'])
    
        for _file_name, module_name in all_files.items():
            if file_name == _file_name:
                self.remove(module_name, file_name)
        
        self.metadata['log'] = filter(lambda x: (x['file_name']!=file_name), 
                                                 self.metadata['log'])     
        self.write_metadata()


    def transform(self, module_name, params):
        '''Run module on pandas DataFrame in memory and update memory state'''
        self.check_mem_data()        
        
        # Initiate log
        log = self.init_log(module_name, 'transform')

        # TODO: catch module errors and add to log
        # Run module on pandas DataFrame 
        self.mem_data = MODULES['transform'][module_name]['func'](self.mem_data, params)
        self.mem_data_info['module_name'] = module_name
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        return log
    




class UserNormalizer(Normalizer):
    def path_to(self, module_name='', file_name=''):
        return self._path_to(NORMALIZE_DATA_PATH, module_name, file_name)
    
class InternalNormalizer(Normalizer):
    def path_to(self, module_name='', file_name=''):
        return self._path_to(NORMALIZE_DATA_PATH, module_name, file_name)
    




if __name__ == '__main__':
    import json
    import logging

    source_file_name = 'source.csv' # 'SIREN_FUI.col' # 'abes.csv'
    user_given_name = 'first_file.csv'

    logging.basicConfig(filename = 'log/preprocess_fields.log', level = logging.DEBUG)
    
    # Create/Load a project
    project_id = "4e8286f034eef40e89dd99ebe6d87f21"
    proj = UserNormalizer(None, create_new=True)
    
    # Upload file to normalize
    file_path = os.path.join('local_test_data', source_file_name)
    with open(file_path) as f:
        proj.upload_init_data(f, source_file_name, user_given_name)


    # Load source data to memory
    proj.load_data(module_name='INIT' , file_name=user_given_name)
    
    
    inferredTypes = proj.infer('inferTypes', params = None)
    
    print('Inferred data types:', inferredTypes)

    # Try transformation
    params = { 'dataTypes': {
    'TEL': 'Téléphone',
    'EMAIL': 'Email',
    'WEB': 'URL',
    'ADPHYSIQUE': 'Voie',
    'VILLE': 'Commune',
    'CDPOSTAL': 'Code Postal',
    'PAYS': 'Pays' } }
    proj.transform('normalize', params)
    
    # Write transformed file
    proj.write_data()
    proj.write_log_buffer(written=True)
    
    # Remove previously uploaded file
    # proj.remove_data('source', 'INIT', 'source.csv')    

    assert False
