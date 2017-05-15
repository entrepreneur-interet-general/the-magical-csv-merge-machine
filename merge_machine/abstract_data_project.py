#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:18:44 2017

@author: leo
"""

import csv
import gc
import os
import time

import pandas as pd

from abstract_project import AbstractProject, NOT_IMPLEMENTED_MESSAGE

from MODULES import MODULES

class AbstractDataProject(AbstractProject):
    '''
    Allows loading and writing of data objects (pandas DataFrames) and 
    anticipates usage of transformations and writing them to log
    '''
    
    def init_log(self, module_name, module_type):
        raise Exception(NOT_IMPLEMENTED_MESSAGE)
        
    def end_log(self, log, error=False):
        '''
        Close a log mesage started with init_log (right after module call)
        '''
        log['end_timestamp'] = time.time()
        log['error'] = error
        return log    
    
    def check_mem_data(self):
        '''Check that there is data loaded in memory'''
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')
        
    def load_data(self, module_name, file_name, nrows=None):
        '''Load data as pandas DataFrame to memory'''
        file_path = self.path_to(module_name, file_name)
        if nrows is not None:
            print('Nrows is : ', nrows)
            self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype=str, nrows=nrows)
        else:
            self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        self.mem_data_info = {'file_name': file_name,
                              'module_name': module_name}
        
    def get_header(self, module_name, file_name):
        file_path = self.path_to(module_name, file_name)
        header = list(pd.read_csv(file_path, encoding='utf-8', nrows=0).columns)
        return header
    
    def _get_sample(self, module_name, file_name, row_idxs=range(5), 
                   columns=None, drop_duplicates=True):
        '''
        Returns a dict with the selected rows (including header)
        
        INPUT:
            - module_name: module where data is stored
            - file_name: name of file to load from
            - row_idxs: list of indexes of rows to show (as in CSV file, 
                        0 is header and is mandatory as first element of list)
            - columns: list of columns to display. By default, all are shown
            - drop_duplicates: wheather or not to show duplicate rows
        '''
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
        
    def get_sample(self, sampler_module_name, params, sample_params):
        '''
        Returns an interesting sample for the data and config at hand.
        
        NB: This is here for uniformity with transform and infer
        
        INPUT:
            - sampler_module_name: name of sampler function (None for first N
                                                             rows)
            - params: inference params to send to sampler to help with selection
            - sample_params: parameters concerning the size of output etc.
        OUTPUT:
            - sample
        
        '''
        self.check_mem_data()
        
        if sampler_module_name is not None:
            sample_ilocs = MODULES['sample'][sampler_module_name]['func'](self.mem_data, 
                                                              params, sample_params)
        else:
            sample_ilocs = sample_params.get('sample_ilocs', range(5))
            

        cols_to_display = sample_params.get('cols_to_display', self.mem_data.columns)
        sub_tab = self.mem_data.iloc[sample_ilocs].loc[:, cols_to_display]

        
        if sample_params.get('drop_duplicates', True):
            sub_tab.drop_duplicates(inplace=True)

        # Replace missing values
        sub_tab.fillna('', inplace=True)
        
        #        sample = self._get_sample(module_name, file_name,
        #                                 row_idxs=[0] + [x+1 for x in sample_ilocs])        
        sample = sub_tab.to_dict('records')    
        return sample
    
    
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
    