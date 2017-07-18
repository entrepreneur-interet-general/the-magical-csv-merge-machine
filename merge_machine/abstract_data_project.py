#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:18:44 2017

@author: leo

AbstractDataProject

METHODS:
    
    - init_active_log(self, module_name, module_type)
    - end_active_log(self, log, error=False)
    - check_mem_data(self)
    - load_data(self, module_name, file_name, nrows=None, columns=None)
    - get_header(self, module_name, file_name)
    - get_sample(self, sampler_module_name, params, sample_params)
    - write_log_buffer(self, written)
    - write_run_info_buffer(self)
    - write_data(self)
    - clear_memory(self)
    - infer(self, module_name, params)

"""
from collections import defaultdict
import csv
import gc
from itertools import tee
import os
import time

import numpy as np
import pandas as pd

from abstract_project import AbstractProject, NOT_IMPLEMENTED_MESSAGE


MINI_PREFIX = 'MINI__'

class AbstractDataProject(AbstractProject):
    '''
    Allows loading and writing of data objects (pandas DataFrames) and 
    anticipates usage of transformations and writing them to log
    '''    
    default_module_log = {'completed': False, 'skipped': False}    
    
    CHUNKSIZE = 2000
    
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 description=None,
                 display_name=None):
        super().__init__(project_id=project_id, 
                          create_new=create_new, 
                          description=description,
                          display_name=display_name)
        # Initiate with no data in memory
        self.mem_data = None
        self.mem_data_info =  dict() # Information on data in memory
        self.run_info_buffer = dict()
        self.log_buffer = [] # List of logs not yet written to metadata.json    
        self.last_written = {}

    def default_log(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def upload_init_data(self, file, file_name, user_given_name=None):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def end_active_log(self, log, error=False):
        '''
        Close a log mesage started with init_active_log (right after module call)
        '''
        log['end_timestamp'] = time.time()
        log['error'] = error
        if not error:
            log['completed'] = True
        return log    

    def init_active_log(self, module_name, module_type):
        '''
        Initiate a log (before a module call). Use end_active_log to complete log message
        '''
        # TODO: change this
        # TODO: look where to load source and ref (linker)
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

    def check_mem_data(self):
        '''Check that there is data loaded in memory'''
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')
        
    def load_data(self, module_name, file_name, nrows=None, columns=None):
        '''
        Load data as pandas DataFrame to memory. Overwritten in normalize
        
        Returns a 2-element tuple:
            - first element is a generator which generates pandas DataFrames
            - second element is a 
        '''
        print('columns', columns)
        file_path = self.path_to(module_name, file_name)
        try:
            if nrows is not None:
                print('Nrows is: ', nrows)
                self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype=str, 
                                    nrows=nrows, usecols=columns, chunksize=self.CHUNKSIZE)
            else:
                self.mem_data = pd.read_csv(file_path, encoding='utf-8', dtype=str, 
                                            usecols=columns, chunksize=self.CHUNKSIZE)
        except:
            import pdb; pdb.set_trace()
        self.mem_data_info = {'file_name': file_name,
                              'module_name': module_name,
                              'nrows': nrows, 
                              'columns': columns}

    def reload(self):
        '''
        Loads data with the parameters 
        '''

    def to_xls(self, module_name, file_name):
        '''
        Takes the file specified by module and file names and writes an xls in 
        the same directory with the same name (changing the file extension).
        
        Columns of the original file will be written in the first sheet.
        Columns containing "__" will be written the second sheet
        
        Use for download only!
        '''
        
        file_path = self.path_to(module_name, file_name)
        
        assert file_name[-4:] == '.csv'
        new_file_name = file_name[:-4] + '.xlsx'
        new_file_path = self.path_to(module_name, new_file_name)
        
        tab = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        
        
        columns_og = [x for x in tab.columns if '__' not in x]
        columns_new = [x for x in tab.columns if '__' in x]
        
        writer = pd.ExcelWriter(new_file_path)
        tab[columns_og].to_excel(writer, 'original_file', index=False)
        tab[columns_new].to_excel(writer, 'normalization', index=False)
        writer.save()        
        return new_file_name


    def get_sample(self, sampler_module_name, module_params, sample_params):
        '''
        Returns an interesting sample for the data and config at hand.
        
        NB: This is here for uniformity with transform and infer
        
        INPUT:
            - sampler_module_name: name of sampler function (None for first N
                                                             rows)
            - module_params: inference params to send to sampler to help with selection
            - sample_params: parameters concerning the size of output etc.
        OUTPUT:
            - sample
        
        '''
        self.check_mem_data()
        
        self.mem_data, tab_gen = tee(self.mem_data)
        part_tab = next(tab_gen)
        
        sample_params.setdefault('randomize', True)
        
        num_rows = sample_params.setdefault('num_rows', min(50, part_tab.shape[0]))
        
        # TODO
        if sample_params['randomize']:
            indexes = np.random.permutation(range(part_tab.shape[0]))[:num_rows]
            sample_params.setdefault('sample_ilocs', indexes)
        else:
            sample_params.setdefault('sample_ilocs', range(num_rows))
        
        # 
        if sampler_module_name is None:
            sampler_module_name = 'standard'
        
        sample_ilocs = []
        if sampler_module_name != 'standard':
            sample_ilocs = self.MODULES['sample'][sampler_module_name]['func'](part_tab, 
                                                              module_params, sample_params)
        
        # If default sampler was selected custom sampler returned no rows
        if not sample_ilocs:
            sample_ilocs = sample_params.get('sample_ilocs', range(5))
         
        # Transform int to range if int is received
        #        if isinstance(sample_ilocs, int):
        #            sample_ilocs = range(sample_ilocs)

        cols_to_display = sample_params.get('cols_to_display', part_tab.columns)
        sub_tab = part_tab.iloc[sample_ilocs].loc[:, cols_to_display]

        
        if sample_params.get('drop_duplicates', True):
            sub_tab.drop_duplicates(inplace=True)

        # Replace missing values
        sub_tab.fillna('', inplace=True)
        
        #        sample = self._get_sample(module_name, file_name,
        #                                 row_idxs=[0] + [x+1 for x in sample_ilocs])        
        sample = sub_tab.to_dict('records')    
        return sample
    
    @staticmethod
    def _is_mini(file_name):
        return file_name[:len(MINI_PREFIX)] == MINI_PREFIX
    
    @staticmethod
    def _og_from_mini(file_name):
        '''Returns the original file name from the MINI version'''
        return file_name[len(MINI_PREFIX):]
    
    def make_mini(self, params):
        '''
        Creates a smaller version of the table in memory. 
        Set mem_data_info and current file to mini
        '''
        # TODO: Change current 
        # TODO: Current for normalize ?        
        self.check_mem_data()
    
        # Set defaults
        sample_size = params.get('sample_size', self.CHUNKSIZE - 1)
        randomize = params.get('randomize', True)       
        new_file_name = MINI_PREFIX + self.mem_data_info['file_name']


        if self.mem_data_info['module_name'] != 'INIT':
            raise Exception('make_mini can only be called on data in memory from the INIT module')
       
        part_tab = next(self.mem_data)
        
        # Only create file if it is larger than sample size
        if part_tab.shape[0] > sample_size:            

            self.clean_after('INIT', new_file_name) # TODO: check module_name for clean_after
            
            # Initiate log
            log = self.init_active_log('INIT', 'transform')  # TODO: hack here: module_name should be 'make_mini'
            
            if randomize:
                sample_index = np.random.permutation(part_tab.index)[:sample_size]
            else:
                sample_index = part_tab.index[:sample_size]
            
            # Replace data in memory
            self.mem_data = (x for x in [part_tab.loc[sample_index, :]])
            
            # Update metadata and log
            self.metadata['has_mini'] = True
            self.mem_data_info['file_name'] = new_file_name
            
            # Create new empty log in metadata # TODO: make class method
            self.metadata['log'][new_file_name] = self.default_log()
        
            log['og_file_name'] = log['file_name']
            log['file_name'] = new_file_name
            
            # TODO: think if transformation should / should not be complete
    
            # Complete log
            log = self.end_active_log(log, error=False) 
                              
            # Update buffers
            self.log_buffer.append(log) # TODO: change for dict
            # TODO: Make sure that run_info_buffer should not be extended
            return log
        
        else:
            self.metadata['has_mini'] = False
            return {}
    
    def write_log_buffer(self, written):
        '''
        Appends log buffer to metadata, writes metadata and clears log_buffer.
        
        INPUT: 
            - written: weather or not the data was written
        '''
        if not self.log_buffer:
            # TODO: Put warning here
            pass
            #            raise Exception('No log buffer: no operations were executed since \
            #                            write_log_buffer was last called')

        # Indicate if any data was written
        if written:
            for log in self.log_buffer[::-1]:
                assert log['module_type'] in ['infer', 'transform', 'link']
                if log['module_type'] in ['transform', 'link']:
                    log['written'] = True
                    self.metadata['last_written'] = {
                                                    'module_name': log['module_name'], 
                                                    'file_name': log['file_name']
                                                    }
                    break
                       
        # Add buffer to metadata
        for log in self.log_buffer:
            file_name = log['file_name']
            module_name = log['module_name']
            
            #            if file_name not in self.metadata['log']:
            #                raise ValueError('file name {0} was not initialized in log'.format(file_name))
            #            if module_name in self.metadata['log'][file_name]:
            #                raise ValueError('module name {0} was not initialized in log for file {1}'.format(module_name, file_name))
            if file_name is not None: # TODO: burn this heresy
                self.metadata['log'][file_name][module_name].update(log)
        # Write metadata and clear log buffer
        self.write_metadata()
        self.log_buffer = []


    def write_run_info_buffer(self):
        '''
        Appends run info buffer to metadata, writes metadata and clears run_info_buffer.        
        '''
        # TODO: run_info should be file_name aware
        for (module_name, file_name), run_info in self.run_info_buffer.items():
            config_file_name = file_name + '__run_info.json'
            self.upload_config_data(run_info, module_name, config_file_name)
        self.run_info_buffer = dict()
        
    def write_data(self):
        '''Write data stored in memory to proper module'''
        self.check_mem_data()
            
        # Write data
        dir_path = self.path_to(self.mem_data_info['module_name'])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(self.mem_data_info['module_name'], 
                                 self.mem_data_info['file_name'])
#        try:
        nrows = 0
        with open(file_path, 'w') as w:
            # Enumerate to know whether or not to write header (i==0)
            try:
                for i, part_tab in enumerate(self.mem_data):
                    print(i)
                    part_tab.to_csv(w, encoding='utf-8', 
                                         index=False,  
                                         header=i==0, 
                                         quoting=csv.QUOTE_NONNUMERIC)
                    nrows += len(part_tab)
                    
            except Exception as e:
                print('At error here:', e)
#        except Exception as e:
#            if os.path.isfile(file_path):
#                os.remove(file_path)
#            raise e

        print('Wrote to ', file_path)
        self.write_log_buffer(True)
        self.write_run_info_buffer()


        if nrows == 0:
            raise Exception('No data was written, make sure you loaded data before'
                           + ' calling write_data')
        return nrows

        
    def clear_memory(self):
        '''Removes the table loaded in memory'''
        self.mem_data = None
        self.mem_data_info = dict()
        gc.collect()
        
        
    def infer(self, module_name, params):
        '''
        Runs the module on pandas DataFrame data in memory and 
        returns answer + writes to appropriate location
        '''
        
        # Check that memory is loaded (if necessary)
        if (params is not None) and params.get('NO_MEM_DATA', False):
            data = None
        else:
            self.check_mem_data()
            self.mem_data, tab_gen = tee(self.mem_data)
            data = pd.concat(tab_gen)
            
        # Initiate log
        log = self.init_active_log(module_name, 'infer')
        
        # We duplicate the generator to load a full version of the table and
        # while leaving self.mem_data unchanged
        infered_params = self.MODULES['infer'][module_name]['func'](data, params)
        
        # Write result of inference
        module_to_write_to = self.MODULES['infer'][module_name]['write_to']
        self.upload_config_data(infered_params, module_to_write_to, 'infered_config.json')
        
        # Update log buffer
        self.log_buffer.append(log)     
        
        return infered_params
    
    @staticmethod
    def count_modifications(modified):
        '''
        Counts the number of modified values per column
        
        INPUT:
            - modified: pandas DataFrame with booleans to indicate if the value
                        was modified
        OUTPUT:
            - mod_count: dictionary with keys: the columns of the dataframe
                        and values: the number of "True" per column
        '''
        mod_count = modified.sum().to_dict()
        return mod_count
    
    @staticmethod
    def add_mod(mod_count, new_mod_count):
        '''
        Updates the modification count
        '''
        for col, count in new_mod_count.items():
            mod_count[col] += count
        return mod_count        
    
    def run_transform_module(self, module_name, partial_data, params):
        '''
        Runs the selected module on the dataframe in partial_data and stores
        modifications in the run_info buffer
        '''
        print('At module ', module_name)
        # Apply module transformation
        valid_columns = [col for col in partial_data if '__' not in col] # TODO: test on upload      
        created_columns = [col for col in partial_data if '__' in col]
        old_modified = partial_data[created_columns]
        
        new_partial_data, modified = self.MODULES['transform'][module_name] \
                                    ['func'](partial_data[valid_columns], params)
                                    
        for col in created_columns:
            new_partial_data.loc[:, col] = old_modified[col]
        
        # Add modifications tracker to data
        modified.columns = [col + '__MODIFIED' for col in modified.columns]
        
        for col in modified.columns:
            if col in new_partial_data:
                new_partial_data[col] = new_partial_data[col] | modified[col]
            else:
                new_partial_data[col] = modified[col]

        # Store modificiations in run_info_buffer
        self.run_info_buffer[(module_name, self.mem_data_info['file_name'])]['mod_count'] = \
            self.add_mod(self.run_info_buffer[(module_name, self.mem_data_info['file_name'])]['mod_count'], \
                         self.count_modifications(modified))
        return new_partial_data
    
    def transform(self, module_name, params):
        '''
        Run module on pandas DataFrame in memory and update memory state.
        /!\ DATA IS CLEANED WHEN transform IS CALLED
        '''
        self.check_mem_data()
        self.clean_after(module_name, self.mem_data_info['file_name'])
        
        # Initiate log
        log = self.init_active_log(module_name, 'transform')

        # Initiate run_info
        run_info = dict()
        run_info['file_name'] = self.mem_data_info['file_name']
        run_info['module_name'] = module_name
        run_info['params'] = params
        run_info['mod_count'] = defaultdict(int)
        self.run_info_buffer[(module_name, run_info['file_name'])] = run_info

        # TODO: catch module errors and add to log
        # Run module on pandas DataFrame 
        self.mem_data = (self.run_transform_module(module_name, data, params) \
                                                 for data in self.mem_data)
        self.mem_data_info['module_name'] = module_name

        # Complete log
        log = self.end_active_log(log, error=False)
                          
        # Add time to run_info (# TODO: is this the best way?) 
#        run_info['start_timestamp'] = log['start_timestamp']
#        run_info['end_timestamp'] = log['end_timestamp']
        
        # Update buffers
        self.log_buffer.append(log)
        return log, run_info