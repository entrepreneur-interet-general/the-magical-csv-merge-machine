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

from abstract_project import AbstractProject, NOT_IMPLEMENTED_MESSAGE
from infer_nan import infer_mvs, replace_mvs, sample_mvs
from preprocess_fields_v3 import inferTypes, normalizeValues
from results_analysis import results_analysis

MODULES = {
        'transform':{
                    'INIT': {
                                'desc': 'Initial upload (cannot be called)'                            
                            },
                    'replace_mvs': {
                                'func': replace_mvs,
                                'desc': replace_mvs.__doc__
                            },
                    'normalize': {
                                'func':  normalizeValues,
                                'desc': normalizeValues.__doc__
                            }
                    },
        'infer':{
                'infer_mvs': {
                                'func': infer_mvs,
                                'write_to': 'replace_mvs',
                                'desc': infer_mvs.__doc__
                            },
                'inferTypes': {
                                'func': inferTypes,
                                'write_to': 'normalize',
                                'desc': inferTypes.__doc__
                            },
                'results_analysis': {
                                'func': results_analysis,
                                'write_to': 'results_analysis',
                                'desc': results_analysis.__doc__
                            }
                },
                
        'sample': {
                'sample_mvs': {
                        'func': sample_mvs,
                        'desc': sample_mvs.__doc__
                        }
                }
        }



class Normalizer(AbstractProject):
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

    def select_file(self, file_name, internal=False, project_id=None):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)
    
#==============================================================================
# Actual class
#==============================================================================   

    def __init__(self, project_id=None, create_new=False, description=''):
        super().__init__(project_id, create_new, description)
    

    def path_to(self, data_path, module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        if module_name is None:
            module_name = ''
        if file_name is None:
            file_name = ''
        
        path = os.path.join(data_path, self.project_id, module_name, file_name)

        return os.path.abspath(path)    

    def create_metadata(self, description=''):
        metadata = dict()
        metadata['description'] = description
        metadata['log'] = []
        metadata['project_id'] = self.project_id
        metadata['timestamp'] = time.time()
        metadata['user_id'] = '__ NOT IMPLEMENTED'
        return metadata   

    def init_log(self, module_name, module_type):
        '''
        Initiate a log (before a module call). Use end_log to complete log message
        '''
        assert module_type in ['transform', 'infer']
        log = { 
                # Data being modified
               'file_name': self.mem_data_info.get('file_name', None), 
               'origin': self.mem_data_info.get('module_name', None),
               
                # Modification at hand                        
               'module_name': module_name, # Module to be executed
               'module_type': module_type, # Type (transform, infer, or dedupe)
               'start_timestamp': time.time(),
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False
               }
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
            - before_module: Looks for file that was written in a module previous or
              or equal to before_module (in the order defined by MODULE_ORDER)
            
        OUTPUT:
            - (module_name, file_name)
        '''
        
        MODULE_ORDER = ['INIT', 'replace_mvs']
        
        for module_from_loop in MODULE_ORDER:
            assert (module_from_loop in MODULES['transform'])
        
        previous_modules = {MODULE_ORDER[i]: MODULE_ORDER[:i+1] for i in range(len(MODULE_ORDER))}
    
    
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

    def check_mem_data(self):
        '''Check that there is data loaded in memory'''
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')        

    def delete_project(self):
        '''Deletes entire folder containing the project'''
        path_to_proj = self.path_to()
        shutil.rmtree(path_to_proj)
    
    def upload_init_data(self, file, file_name, user_given_name=None):
        # TODO: deal with og_file_name, file_id, display_name, user_given_name
        
        """
        Upload and write source or reference to the project. Tables will
        be added to the "INIT" module.
        
        The file will be re-coded in utf-8 with a "," separator. Also, chars
        specified in CHARS_TO_REPLACE will be replaced by "_" in the header.
        """
        CHARS_TO_REPLACE = [' ', ',', '.', '(', ')', '\'', '\"']
        ENCODINGS = ['utf-8', 'windows-1252']
        SEPARATORS = [',', ';', '\t']
                
        # Check that file is not already present
        self.mem_data_info = {'file_name': file_name,
                              'module_name': 'INIT'}
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

        # TODO: add error if same column names

        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)
        
        # write data and log
        self.write_data()
        self.write_log_buffer(written=True)
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
        
        
    def load_data(self, module_name, file_name):
        '''Load data as pandas DataFrame to memory'''
        file_path = self.path_to(module_name, file_name)
        self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        self.mem_data_info = {'file_name': file_name,
                              'module_name': module_name}
        
    def get_header(self, module_name, file_name):
        file_path = self.path_to(module_name, file_name)
        header = list(pd.read_csv(file_path, encoding='utf-8', nrows=0).columns)
        return header
    
    def get_sample(self, module_name, file_name, row_idxs=range(5), 
                   columns=None, drop_duplicates=True):
        '''Returns a dict with the selected rows (including header)'''
        file_path = self.path_to(module_name, file_name)
        
        if row_idxs is not None:
            if row_idxs[0] != 0:
                raise Exception('Row selection for samples (row_idxs) should\
                                include header (row_idxs[0]==0)')
                
            # Load the right amount of rows
            max_rows = max(row_idxs)    

            tab = pd.read_csv(file_path, encoding='utf-8', dtype=str, 
                              usecols=columns, nrows=max_rows)
            
            # row_idxs counts lines in csv including header --> de-increment
            try:
                tab = tab.iloc[[x - 1 for x in row_idxs[1:]], :]
            except:
                import pdb
                pdb.set_trace()
        else:
            tab = pd.read_csv(file_path, encoding='utf-8', dtype=str, usecols=columns)
        
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
        dir_path = self.path_to(self.mem_data_info['module_name'])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(self.mem_data_info['module_name'], 
                                 self.mem_data_info['file_name'])
        self.mem_data.to_csv(file_path, encoding='utf-8', index=False, 
                             quoting=csv.QUOTE_NONNUMERIC)
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
        self.upload_config_data(infered_params, module_to_write_to, 'infered_config.json')
        
        # Update log buffer
        self.log_buffer.append(log)     
        
        return infered_params
    

    def gen_sample(self, paths, module_name, params, sample_params):
        '''
        Returns an interesting sample for the data and config at hand.
        
        NB: This is here for uniformity with transform and infer
        '''
        module_name = paths['module_name']
        file_name = paths['file_names']
        
        sample_ilocs = MODULES['sample'][module_name]['func'](self.mem_data, params, sample_params)
        
        sample = self.get_sample(module_name, file_name,
                                 row_idxs=[0] + [x+1 for x in sample_ilocs])        
        return sample
