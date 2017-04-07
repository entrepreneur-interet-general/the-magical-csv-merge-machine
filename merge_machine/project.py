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

    - Follow module writing order (MODULE_ORDER)

    - Dedupe: Numbers predicate ? Lycée Acajou 1	LYCEE POLYVALENT ACAJOU 2

TODO:
    - Store user params
    - Deal with log for merge 
    
    - Create Data Class ?
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
        'transform':{
                    'INIT': {
                            'desc': 'Initial upload (cannot be called)'                            
                            },
                    'replace_mvs': {
                                    'func': replace_mvs,
                                    'desc': 'Replace strings that represent missing values'
                                    }
                    },
        'infer':{
                'infer_mvs': {
                            'func': infer_mvs,
                            'write_to': 'replace_mvs',
                            'desc': 'Infer values that represent missing values'
                            }
                },
        'link': {
                'dedupe_linker': {
                            'func': dedupe_linker,
                            'desc': 'Link CSV files'
                            }
                }
        }

        

def gen_proj_id():
    '''Generate unique non-guessable string for project ID'''
    unique_string = str(time.time()) + '_' + str(random.random())
    h = hashlib.md5()
    h.update(unique_string.encode('utf-8'))
    project_id = h.hexdigest()
    return project_id

    
def get_arb(dir_path):
    '''Return arborescence as dict'''
    to_return = dict()
    for name in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, name)):
            to_return[name] = get_arb(os.path.join(dir_path, name))
        else:
            to_return[name] = {}
    return to_return



NOT_IMPLEMENTED_MESSAGE = 'NOT IMPLEMENTED in abstract class'

class Project():
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
# Methods that should be implemented in children
#==============================================================================

    def check_file_role(self, file_role):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)

    def path_to(self, file_role='', module_name='', file_name=''):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)

    def create_metadata(self, description=''):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)    

    def select_file(self, file_role, file_name, internal=False, project_id=None):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)
    
#==============================================================================
# Actual class
#==============================================================================

    def __init__(self, project_id=None, create_new=False, description=''):
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
    

    def init_log(self, module_name, module_type):
        '''
        Initiate a log (before a module call). Use end_log to complete log message
        '''
        assert module_type in ['transform', 'infer', 'link']
        log = { # Data being modified
               'file_name': self.mem_data_info.get('file_name', None), 
               'origin': self.mem_data_info.get('module_name', None),
               'file_role': self.mem_data_info.get('file_role', None),
                # Modification at hand                        
               'module_name': module_name, # Module to be executed
               'module_type': module_type, # Type (transform, infer, or dedupe)
               'start_timestamp': time.time(),
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False}
        return log
        
    def end_log(self, log, error=False):
        '''
        Close a log mesage started with init_log (right after module call)
        '''
        log['end_timestamp'] = time.time()
        log['error'] = error
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


    def get_arb(self):
        '''List directories and files in project'''
        path_to_proj = self.path_to()
        return get_arb(path_to_proj)
    

    def _list_files(self, extensions=['.csv']):
        '''
        Lists csv files (from data) in data directory and presents a list of modules in 
        which they are present. You can combine this with get_last_written
        '''
        def is_dir(root_path, x):
            return os.path.isdir(os.path.join(root_path, x))
        
        all_files = dict()
        root_path = self.path_to()
        for file_role in filter(lambda x: is_dir(root_path, x), os.listdir(root_path)):
            all_files[file_role] = dict()
            sub_root_path = self.path_to(file_role=file_role)
            for _dir in filter(lambda x: is_dir(sub_root_path, x), os.listdir(sub_root_path)):
                for file_name in os.listdir(os.path.join(sub_root_path, _dir)):
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
            


    def get_last_written(self, file_role=None, module_name=None, file_name=None, 
                         before_module=None):
        '''
        Return info on data that was last successfully written (from log)
        
        INPUT:
            - file_role: filter on file role (last data with given file_role)
            - module_name: filter on given module
            - file_name: filter on file_name
            - before_module: Looks for file that was written in a module previous or
              or equal to before_module (in the order defined by MODULE_ORDER)
            
        OUTPUT:
            - (file_role, module_name, file_name)
        '''
        
        MODULE_ORDER = ['INIT', 'replace_mvs', 'dedupe_linker']
        
        for module_from_loop in MODULE_ORDER:
            assert (module_from_loop in MODULES['transform']) \
                or (module_from_loop in MODULES['link'])
            
        
        previous_modules = {MODULE_ORDER[i]: MODULE_ORDER[:i+1] for i in range(len(MODULE_ORDER))}
        
        self.check_file_role(file_role)
    
        for log in self.metadata['log'][::-1]:
            if (not log['error']) and log['written'] \
                      and ((file_role is None) or (log['file_role'] == file_role)) \
                      and ((module_name is None) or (log['module_name'] == module_name)) \
                      and ((file_name is None) or (log['file_name'] == file_name)) \
                      and ((before_module is None) or (log['module_name'] in previous_modules[before_module])):                
                break
        else:
            raise Exception('No written data could be found in logs')
        
        file_role = log['file_role']
        module_name = log['module_name']
        file_name = log['file_name']        
        return (file_role, module_name, file_name)

    def path_to_last_written(self, file_role=None, module_name=None, 
                             file_name=None, before_module=None):
        (file_role, module_name, file_name) = self.get_last_written(file_role, 
                                        module_name, file_name, before_module)
        path = self.path_to(file_role, module_name, file_name)
        return path

    def check_mem_data(self):
        '''Check that there is data loaded in memory'''
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')        

    def delete_project(self):
        '''Deletes entire folder containing the project'''
        path_to_proj = self.path_to()
        shutil.rmtree(path_to_proj)
    
    def upload_init_data(self, file, file_role, file_name):
        """
        Upload and write source or reference to the project. Tables will
        be added to the "INIT" module.
        
        The file will be re-coded in utf-8 with a "," separator. Also, chars
        specified in CHARS_TO_REPLACE will be replaced by "_" in the header.
        """
        CHARS_TO_REPLACE = [' ', ',', '.', '(', ')', '\'', '\"']
        ENCODINGS = ['utf-8', 'windows-1252']
        SEPARATORS = [',', ';', '\t']
        
        self.check_file_role(file_role)
        
        # Check that file is not already present
        
        #
        self.mem_data_info = {'file_role': file_role, 
                              'file_name': file_name,
                              'module_name': 'INIT'}
        log = self.init_log('INIT', 'transform')
        
        could_read = False
        for encoding in ENCODINGS:
            for sep in SEPARATORS:
                try:
                    self.mem_data = pd.read_csv(file, sep=sep, encoding=encoding, dtype='unicode')
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

        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        
        # write data and log
        self.write_data()
        self.write_log_buffer(written=True)
        self.clear_memory()
    

    def upload_config_data(self, config_dict, file_role, module_name, file_name):
        '''Will write config file'''
        VALID_FILE_NAMES = ['config.json', 'infered_config.json', 
                             'training.json', 'column_matches.json',
                             'columns_to_return_source.json', 
                             'columns_to_return_ref.json']
        
        if config_dict is None:
            return
        
        if (module_name not in MODULES['link']) and (module_name not in MODULES['transform']) :
            raise Exception('Config files can only be uploaded to module \
                            directories of type link or transform') 
            
        if file_name not in VALID_FILE_NAMES:
            raise Exception('For now you can only upload files named \
                            training.json or column_matches.json')

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

    def remove_all(self, file_name):
        '''
        Remove all occurences of files with a given file_name from the project
        and clears all mentions of this file_name in metadata
        '''
        # TODO: deal with .csv dependency
        all_files = self._list_files(extensions=['.csv'])
        
        for file_role, name_dict in all_files.items():
            for _file_name, module_name in name_dict.items():
                if file_name == _file_name:
                    self.remove(file_role, module_name, file_name)
        
        self.metadata['log'] = filter(lambda x: (x['file_name']!=file_name), 
                                                 self.metadata['log'])     
        self.write_metadata()
        
        
    def load_data(self, file_role, module_name, file_name):
        '''Load data as pandas DataFrame to memory'''
        file_path = self.path_to(file_role, module_name, file_name)
        self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype='unicode')
        self.mem_data_info = {'file_role': file_role, 
                               'file_name': file_name,
                               'module_name': module_name}
        
    def get_header(self, file_role, module_name, file_name):
        file_path = self.path_to(file_role, module_name, file_name)
        header = list(pd.read_csv(file_path, encoding='utf-8', nrows=0).columns)
        return header
    
    def get_sample(self, file_role, module_name, file_name, row_idxs=range(5), 
                   columns=None, drop_duplicates=True):
        '''Returns a dict with the selected rows (including header)'''
        file_path = self.path_to(file_role, module_name, file_name)
        
        if row_idxs is not None:
            if row_idxs[0] != 0:
                raise Exception('Row selection for samples (row_idxs) should\
                                include header (row_idxs[0]==0)')
                
            # Load the right amount of rows
            max_rows = max(row_idxs)    

            tab = pd.read_csv(file_path, encoding='utf-8', dtype='unicode', 
                              usecols=columns, nrows=max_rows)
            
            # row_idxs counts lines in csv including header --> de-increment
            try:
                tab = tab.iloc[[x - 1 for x in row_idxs[1:]], :]
            except:
                import pdb
                pdb.set_trace()
        else:
            tab = pd.read_csv(file_path, encoding='utf-8', dtype='unicode', usecols=columns)
        
        if drop_duplicates:
            tab.drop_duplicates(inplace=True)
        
        # Replace missing values
        tab.fillna('', inplace=True)
        
        return tab.to_dict('records')
        
    
    def write_log_buffer(self, written):
        '''
        Appends log buffer to metadata, writes metadata and clears log_buffer.
        
        INPUT: 
            - written: weather or not the data was written
        
        '''
        if not self.log_buffer:
            raise Exception('No log buffer: no operations were executed since \
                            write_log_buffer was last called')
        
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
        self.log_buffer = []


    def write_data(self):
        '''Write data stored in memory to proper module'''
        self.check_mem_data()
            
        # Write data
        dir_path = self.path_to(self.mem_data_info['file_role'], 
                                self.mem_data_info['module_name'])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(self.mem_data_info['file_role'], 
                                 self.mem_data_info['module_name'], 
                                 self.mem_data_info['file_name'])
        self.mem_data.to_csv(file_path, encoding='utf-8', index=False)
        print('Wrote to ', file_path)

        
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
        self.mem_data = MODULES['transform'][module_name]['func'](self.mem_data, params)
        self.mem_data_info['module_name'] = module_name
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        return log
    

    def infer(self, module_name, params):
        '''
        Runs the module on pandas DataFrame data in memory and 
        returns answer + writes to appropriate location
        '''
        self.check_mem_data()  
        
        # Initiate log
        log = self.init_log(module_name, 'infer')
            
        infered_params = MODULES['infer'][module_name]['func'](self.mem_data, params)
                
        # Write result of inference
        module_to_write_to = MODULES['infer'][module_name]['write_to']
        self.upload_config_data(params, self.mem_data_info['file_role'], \
                                module_to_write_to, 'infered_config.json')


        # Update log buffer
        self.log_buffer.append(log)     
        
        return infered_params
    
    

    

