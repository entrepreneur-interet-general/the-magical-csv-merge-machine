#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:18:44 2017

@author: leo

AbstractDataProject

METHODS:
    
    - _init_active_log(self, module_name, module_type)
    - _end_active_log(self, log, error=False)
    - _check_mem_data(self)
    - load_data(self, module_name, file_name, nrows=None, columns=None)
    - get_header(self, module_name, file_name)
    - get_sample(self, sampler_module_name, params, sample_params)
    - _write_log_buffer(self, written)
    - _write_run_info_buffer(self)
    - write_data(self)
    - clear_memory(self)
    - infer(self, module_name, params)


https://www.elastic.co/guide/en/elasticsearch/reference/current/vm-max-map-count.html
https://www.elastic.co/guide/en/elasticsearch/guide/current/heap-sizing.html
# ES can fail, try increasing heap size in jvm config file: /etc/elasticsearch/jvm.options

"""
from collections import defaultdict
import copy
import gc
import json
import logging
from itertools import tee
import os
import time

import numpy as np
import pandas as pd

from abstract_project import AbstractProject, NOT_IMPLEMENTED_MESSAGE
from es_connection import es, ic
from merge_machine import es_insert

MINI_PREFIX = 'MINI__'

class AbstractDataProject(AbstractProject):
    '''
    Allows loading and writing of data objects (pandas DataFrames) and 
    anticipates usage of transformations and writing them to log
    '''    
    default_module_log = {'completed': False, 'skipped': False}    
    
    CHUNKSIZE = 3000
    
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 description=None,
                 display_name=None, 
                 public=False):
        super().__init__(project_id=project_id, 
                          create_new=create_new, 
                          description=description,
                          display_name=display_name, 
                          public=public)
        # Initiate with no data in memory
        self.mem_data = None
        self.mem_data_info =  dict() # Information on data in memory
        self.run_info_buffer = dict()
        self.log_buffer = [] # List of logs not yet written to metadata.json    
        self.last_written = {}


    def _default_log(self):
        '''Default log for a new file'''
        return {module_name: copy.deepcopy(self.default_module_log) 
                for module_name in self.MODULE_ORDER_log}
    
    def get_last_written(self, module_name=None, file_name=None, 
                         before_module=None):
        '''
        Return info on data that was last successfully written (from log). 
        Optional filters will restrict results to the subset of data matching
        these filters
        
        INPUT:
            - module_name: filter on given module
            - file_name: filter on file_name
            - before_module: (string with module name) Looks for file that was 
                              written in a module previous to before_module 
                              (in the order defined by self.MODULE_ORDER)
            
        OUTPUT:
            - (module_name, file_name)
        '''        
        if (module_name is not None) and (before_module is not None):
            raise Exception('Variables module_name and before_module cannot be \
                            set simultaneously')

        if module_name is not None:
            modules_to_search = [module_name]
        else:        
            previous_modules = {self.MODULE_ORDER[i]: self.MODULE_ORDER[:i] \
                                for i in range(len(self.MODULE_ORDER))}
            previous_modules[None] = self.MODULE_ORDER
            modules_to_search = previous_modules[before_module][::-1]
        
        if file_name is None:
            file_name = self.metadata['last_written']['file_name']
        
        for module_name in modules_to_search:
            log = self.metadata['log'][file_name][module_name]
            if (not log.get('error', False)) and (log.get('written', False)):
                break
        else:
            raise Exception('No written data could be found in logs')
            
        module_name = log['module_name']
        file_name = log['file_name']        
        
        print('LAST WRITTEN:', module_name, file_name, before_module)
        
        return (module_name, file_name)


    def clean_after(self, module_name, file_name, delete_current_module=True):
        '''
        Removes all occurences of file (file_name) and transformations
        at and after the given module (module_name) based on the order in 
        self.MODULE_ORDER
        
        INPUT:
            - module_name
            - file_name
            - delete_current_module: also delete the file in module_name
        '''
        if file_name not in self.metadata['log']:
            logging.warning('File {0} was not found in logs'.format(file_name))
            return
        
        start_idx = self.MODULE_ORDER_log.index(module_name) + int(not delete_current_module)
        for iter_module_name in self.MODULE_ORDER_log[start_idx:]:            
            # module_log = self.metadata['log'][file_name]
            # TODO: check skipped, written instead of try except            
            try:
                self._remove(iter_module_name, file_name)   
                # os.remove(file_path)
                logging.warning('clean_after: Removed {0} / {1}'.format(iter_module_name, file_name))
            except FileNotFoundError:
                pass
            
            try:
                # Retrieve skip info # TODO: rather ugly
                skip = self.metadata['log'][file_name][iter_module_name].get('skipped', 'False')
                self.metadata['log'][file_name][iter_module_name] = copy.deepcopy(self.default_module_log)
                self.metadata['log'][file_name][iter_module_name]['skipped'] = skip
            except:
                pass
            self._write_metadata()


    def _get_header(self, module_name, file_name):
        #TODO: this is a cheap fix for a specific use. DO NOT USE
        file_path = self.path_to(module_name, file_name)
        return list(pd.read_csv(file_path, encoding='utf-8').columns)

    def upload_init_data(self, file, file_name, user_given_name=None):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def _end_active_log(self, log, error=False):
        '''    
        Close a log mesage started with _init_active_log (right after module call)
        and append to log buffer
        
        INPUT:
            - log: a log object created by _init_active_log
            - error: whether or not there was an error during execution
        '''
        log['end_timestamp'] = time.time()
        log['error'] = error
        if not error:
            log['completed'] = True
        self.log_buffer.append(log)# TODO: change for dict
        return log    

    def _init_active_log(self, module_name, module_type):
        '''
        Initiate a log (before a module call). Use _end_active_log to complete 
        log message
        
        INPUT:
            - module_name:
            - module_type: 'transform', 'infer', or 'link'
        '''
        # TODO: change this
        log = { 
                # Data being modified
               'file_name': self.mem_data_info.get('file_name'), 
               'origin': self.mem_data_info.get('module_name'),
               
                # Modification at hand                        
               'module_name': module_name, # Module to be executed
               'module_type': module_type, # Type (transform, infer, or link)
               'start_timestamp': time.time(),
               'end_timestamp': None, 'error':None, 'error_msg':None, 'written': False
               }
        return log
    
    def set_skip(self, module_name, file_name, skip_value):
        '''Sets the "skipped value to True in log'''
        
        # Set skipped for MINI version / full version of file
        file_names = [file_name]
        if self.metadata['has_mini']:
            if self._is_mini(file_name):
                file_names.append(self._og_from_mini(file_name))
            else:
                file_names.append(self._mini_from_og(file_name))
    
        # Set values
        for file_name in file_names:
            self.metadata['log'][file_name][module_name]['skipped'] = skip_value
        self._write_metadata()
        
    def _check_mem_data(self):
        '''Check that there is data loaded in memory'''
        if self.mem_data is None:
            raise Exception('No data in memory: use `load_data` (reload is \
                        mandatory after dedupe)')
    
    def _static_load_data(self, file_path, nrows=None, columns=None): 
        '''
        Load a pandas DataFrame into memory from a csv in file_path. dtype will 
        be str by default except for some column names with '__'. This function 
        is intended to be used to read results of normalization and/or matching.
        Only the first rows will be read (the number of rows being self.CHUNKSIZE)
        
        INPUT:
            file_path: path to csv file
            nrows: number of rows to read
            columns: columns to load
        
        OUTPUT:
            tab: pandas DataFrame
        '''
        if columns is None:
            columns = pd.read_csv(file_path, encoding='utf-8', dtype=str, 
                                    nrows=0, usecols=columns).columns
        def choose_dtype(col):
            '''Load the correct type according to the column name'''
            if '__MODIFIED' in col:
                return bool
            elif col == '__CONFIDENCE':
                return float
            else:
                return str
            
        dtype = {col: choose_dtype(col) for col in columns}

        if nrows is not None:
            logging.debug('Nrows is: {0}'.format(nrows))
            tab = pd.read_csv(file_path, encoding='utf-8', dtype=dtype, 
                                nrows=nrows, usecols=columns, chunksize=self.CHUNKSIZE)
        else:
            tab = pd.read_csv(file_path, encoding='utf-8', dtype=dtype, 
                                        usecols=columns, chunksize=self.CHUNKSIZE)

        return tab
        
    def load_data(self, module_name, file_name, nrows=None, columns=None):
        '''
        Load data as pandas DataFrame to memory. Overwritten in normalize
        
        Creates two properties for self:
            - mem_data: is a generator which generates pandas DataFrames
            - mem_data_info: contains information on the data currently in memory
            
        INPUT:
            - module_name:
            - file_name:
            - nrows: how many rows to read from the file
            - columns: what columns to load
        '''
        logging.debug('Columns selected to load are: {0}'.format(columns))

        file_path = self.path_to(module_name, file_name)     
        self.mem_data = self._static_load_data(file_path, nrows, columns)
        self.mem_data_info = {'file_name': file_name,
                              'module_name': module_name,
                              'nrows': nrows, 
                              'columns': columns,
                              'data_was_transformed': False}

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
        
        INPUT:
            - module_name:
            - file_name:
        '''
        raise DeprecationWarning('Excel download currently not supported due'\
                                 'to potential memory issues with large files')
        
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
        self._check_mem_data()
        
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
        '''Does name match with that of a "mini" file'''
        return file_name[:len(MINI_PREFIX)] == MINI_PREFIX
    
    @staticmethod
    def _og_from_mini(file_name):
        '''Returns the original file name from the MINI version'''
        return file_name[len(MINI_PREFIX):]
    
    @staticmethod
    def _mini_from_og(file_name):
        '''Returns the name for the MINI version from the original name'''
        return MINI_PREFIX + file_name
    
    def make_mini(self, params):
        '''
        Creates a smaller version of the table in memory. 
        Set mem_data_info and current file to mini
        
        INPUT:
            - params:
                sample_size: Number of rows to include in the "MINI" file
                randomize: chose sample_size elements randomly from the file in
                            memory
        '''
        # TODO: Change current 
        # TODO: Current for normalize ?        
        self._check_mem_data()
    
        # Set defaults
        sample_size = params.get('sample_size', self.CHUNKSIZE - 1)
        randomize = params.get('randomize', True)       
        new_file_name = self._mini_from_og(self.mem_data_info['file_name'])

        
        if self.mem_data_info['module_name'] != 'INIT':
            raise Exception('make_mini can only be called on data in memory from the INIT module')
       
        part_tab = next(self.mem_data)
        
        # Only create file if it is larger than sample size
        if part_tab.shape[0] > sample_size:            
            logging.info('Running make_mini')
            self.clean_after('INIT', new_file_name) # TODO: check module_name for clean_after
            
            # Initiate log
            log = self._init_active_log('INIT', 'transform')  # TODO: hack here: module_name should be 'make_mini'
            
            if randomize:
                sample_index = np.random.permutation(part_tab.index)[:sample_size]
            else:
                sample_index = part_tab.index[:sample_size]
            
            # Replace data in memory
            self.mem_data = (x for x in [part_tab.loc[sample_index, :]])
            
            # Update metadata and log
            self.metadata['has_mini'] = True
            self.mem_data_info['file_name'] = new_file_name
            
            # Create new empty log in metadata
            self.metadata['log'][new_file_name] = self._default_log()
        
            log['og_file_name'] = log['file_name']
            log['file_name'] = new_file_name
            
            # TODO: think if transformation should / should not be complete
    
            # Complete log
            log = self._end_active_log(log, error=False) 
        
            # TODO: Make sure that run_info_buffer should not be extended
            return log
        
        else:
            self.metadata['has_mini'] = False
            return {}
    
    def _write_log_buffer(self, written):
        '''
        Appends log buffer to metadata, writes metadata and clears log_buffer.
        
        INPUT: 
            - written: whether or not the data was written
        '''
        if not self.log_buffer:
            # TODO: Put warning here
            pass
            #            raise Exception('No log buffer: no operations were executed since \
            #                            _write_log_buffer was last called')

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
        self._write_metadata()
        self.log_buffer = []


    def _write_run_info_buffer(self):
        '''
        Appends run info buffer to metadata, writes metadata and clears run_info_buffer.        
        '''
        # TODO: run_info should be file_name aware
        for (module_name, file_name), run_info in self.run_info_buffer.items():
            config_file_name = file_name + '__run_info.json'
            self.upload_config_data(run_info, module_name, config_file_name)
        self.run_info_buffer = dict()
        
    def write_data(self):
        '''
        Write data stored in memory to proper module.
        
        The data will be taken from self.mem_data_info. It will be written to 
        the path corresponding to module self.mem_data_info['module_name'] 
        and file self.mem_data_info['file_name'].
        '''
        self._check_mem_data()
        
        # Write data
        dir_path = self.path_to(self.mem_data_info['module_name'])
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)        
        file_path = self.path_to(self.mem_data_info['module_name'], 
                                 self.mem_data_info['file_name'])

        
        # TODO: move this. This was done to avoid concat with init when no changes were made
        nrows = 0
        if os.path.isfile(file_path):
            logging.warning('File {0} already exists. Will not be re-written')

        else:
            print(self.mem_data_info)
            with open(file_path, 'w') as w:
                # If any data was transformed: write the transformed data
                # if self.mem_data_info['data_was_transformed']:
                try:
                    # Enumerate to know whether or not to write header (i==0)
                    for i, part_tab in enumerate(self.mem_data):
                        logging.debug('At part {0}'.format(i))
                        part_tab.to_csv(w, encoding='utf-8', 
                                             index=False,  
                                             header=i==0)
                                             #quoting=csv.QUOTE_NONNUMERIC)
                        nrows += len(part_tab)
                        
                except KeyboardInterrupt as e:
                    logging.error(e)

            if nrows == 0:
                raise Exception('No data was written, make sure you loaded data before'
                               + ' calling write_data')

        # TODO: check if valid also when no changes are made...
        logging.info('Wrote to: {0}'.format(file_path))
        self._write_log_buffer(written=True)
        self._write_run_info_buffer()

        # Reset data_was_transformed
        self.mem_data_info['data_was_transformed'] = False
    

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
            self._check_mem_data()
            self.mem_data, tab_gen = tee(self.mem_data)
            data = next(tab_gen) # Selecting the first chunk of data to perform inference
            valid_columns = [col for col in data if '__MODIFIED' not in col]
            data = data[valid_columns]
            
        # Initiate log
        log = self._init_active_log(module_name, 'infer')
        
        # We duplicate the generator to load a full version of the table and
        # while leaving self.mem_data unchanged
        infered_params = self.MODULES['infer'][module_name]['func'](data, params)
        del data
    
        # Write result of inference
        module_to_write_to = self.MODULES['infer'][module_name]['write_to']

        self.upload_config_data(infered_params, module_to_write_to, 'infered_config.json')
        
        # Update log buffer
        self._end_active_log(log, error=False)  
        
        return infered_params
    
    @staticmethod
    def _count_modifications(modified):
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
    def _add_mod(mod_count, new_mod_count):
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
        logging.info('At module: {0}'.format(module_name))
        # Apply module transformation
        valid_columns = [col for col in partial_data if '__MODIFIED' not in col]     
        modified_columns = [col for col in partial_data if '__MODIFIED' in col]
        old_modified = partial_data[modified_columns]
        
        new_partial_data, modified = self.MODULES['transform'][module_name] \
                                    ['func'](partial_data[valid_columns], params)

        # Store modificiations in run_info_buffer
        self.run_info_buffer[(module_name, self.mem_data_info['file_name'])]['mod_count'] = \
            self._add_mod(self.run_info_buffer[(module_name, self.mem_data_info['file_name'])]['mod_count'], \
                         self._count_modifications(modified))

        modified.columns = [col + '__MODIFIED' for col in modified.columns]
        for col in modified.columns:
            if col in old_modified.columns:
                new_partial_data.loc[:, col] = old_modified.loc[:, col] | modified.loc[:, col]
            else:
                new_partial_data[col] = modified[col]
        return new_partial_data
    
    def transform(self, module_name, params):
        '''
        Run module on pandas DataFrame in memory and update memory state.
        /!\ DATA IS CLEANED WHEN transform IS CALLED
        '''
        self._check_mem_data()
        self.clean_after(module_name, self.mem_data_info['file_name'])
        
        # Initiate log
        log = self._init_active_log(module_name, 'transform')

        # Initiate run_info
        run_info = dict()
        run_info['file_name'] = self.mem_data_info['file_name']
        run_info['module_name'] = module_name
        run_info['params'] = params
        run_info['mod_count'] = defaultdict(int)
        self.run_info_buffer[(module_name, run_info['file_name'])] = run_info

        # TODO: catch module errors and add to log
        # Run module on pandas DataFrame 
        logging.info('Running transform: {0}'.format(module_name))
        self.mem_data = (self.run_transform_module(module_name, data, params) \
                                                 for data in self.mem_data)
        self.mem_data_info['module_name'] = module_name
        self.mem_data_info['data_was_transformed'] = True

        # Complete log
        log = self._end_active_log(log, error=False)
        
        return log, run_info
    
class ESAbstractDataProject(AbstractDataProject):
    es_insert_chunksize = 40000
    es = es
    ic = ic

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.index_name = self.project_id
    
    def fetch_by_id(self, size=5, from_=0):
        '''For an indexed table'''
        
        ids = range(from_, from_+size)
        
        bulk = ''
        for id_ in ids:
            bulk += json.dumps({"index" : self.index_name}) + '\n'
            bulk += json.dumps({"query" : {"match" : {"_id": id_}}, "size": 1}) + '\n'
            
        res = es.msearch(bulk)
        return res     
    
    def create_index(self, ref_path, columns_to_index, force=False):
        '''
        INPUT:
            - ref_path: path to the file to index
            - columns_to_index
        '''
        #        import http
        #        http.client._MAXHEADERS = 1000
        
        testing = True      
        
        ref_gen = pd.read_csv(ref_path, 
                          usecols=columns_to_index.keys(),
                          dtype=str, chunksize=self.es_insert_chunksize)
        
        if self.has_index() and force:
            ic.delete(self.index_name)
            
        if not self.has_index():
            logging.info('Creating new index')
            log = self._init_active_log('INIT', 'transform')
            
            index_settings = es_insert.gen_index_settings(columns_to_index)
            
            logging.warning('Creating index')
            logging.warning(index_settings)
            self.ic.create(es, self.index_name, body=json.dumps(index_settings))    
            logging.warning('Inserting in index')
            es_insert.index(ref_gen, self.index_name, testing)
        
            log = self._end_active_log(log, error=False)
        else:
            logging.info('Index already exists')
        self._write_log_buffer(written=False)
    
    def delete_index(self):
        return ic.delete(self.index_name)
        
    def has_index(self):
        return ic.exists(self.index_name)     
    
    def index_is_complete(self):
        pass
        #TODO: Check if thing is thinged
        
    def gen_default_columns_to_index(self, for_linking=True, columns_to_index=None):
        if for_linking:
            analyzers = {'french', 'whitespace', 'integers', 'city', 'n_grams'} # TODO: from config
        else:
            analyzers = {}
        
        # If columns_to_index is already dict
        if isinstance(columns_to_index, dict):
            return columns_to_index            
        
        # If arguments are not valid 
        if (not for_linking) and (columns_to_index is None):
            raise ValueError('if NOT for_linking, you should specify the columns_to_index')
        
        # If a list of columns to analyze is passed
        if isinstance(columns_to_index, list):
            columns_to_index = {col: analyzers for col in columns_to_index}
        
        # If columns_to_index is None (and project is link), fetch from 
        # TODO: cheap fix, move to linker
        if for_linking and (columns_to_index is None):
            column_tracker = self.metadata['column_tracker']
            columns_to_index = {col: analyzers if col in column_tracker['selected'] \
                                else {} for col in column_tracker['original']}
        
        return columns_to_index
        
    
    def delete_project(self):
        '''Deletes entire folder containing the project'''
        if self.has_index():
            self.delete_index()
        super().delete_project()
