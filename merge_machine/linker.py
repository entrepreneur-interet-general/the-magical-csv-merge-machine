#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:46:00 2017

@author: leo
"""

from abstract_project import AbstractProject
from normalizer import InternalNormalizer, UserNormalizer

from MODULES import MODULES

class Linker(AbstractProject):
    
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 display_name=None,
                 description=None):
        
        super().__init__(project_id, create_new, display_name, description)

    def check_file_role(self, file_role):
        if file_role not in ['ref', 'source']:
            raise Exception('file_role should be either "source" or "ref"')

    def check_select(self):
        '''Check that a source and referential were selected'''
        for file_role in ['source', 'ref']:
            if self.metadata['current'][file_role] is None:
                raise Exception('{0} is not defined for this linking project'.format(file_role))

    def create_metadata(self, description=''):
        metadata = super().create_metadata()
        metadata['current'] = {'source': None, 'ref': None} # {'source': {internal: False, project_id: "ABC123", file_name: "source.csv.csv"}, 'ref': None}
        return metadata   

    def add_col_matches(self, column_matches):
        '''column_matches is a json file as list of dict of list'''
        # TODO: add checks on file
        self.upload_config_data(column_matches, 'link', 'dedupe_linker', 'column_matches.json')

    def read_col_matches(self):
        config = self.read_config_data('link', 'dedupe_linker', 'column_matches.json')
        if not config:
            config = []
        return config

    def add_col_certain_matches(self, column_matches):
        '''column_matches is a json file as list of dict of list'''
        # TODO: add checks on file
        self.upload_config_data(column_matches, 'link', 'dedupe_linker', 'column_certain_matches.json')

    def read_col_certain_matches(self):
        config = self.read_config_data('link', 'dedupe_linker', 'column_certain_matches.json')
        if not config:
            config = []
        return config    
    
    def add_cols_to_return(self, file_role, columns):
        '''
        columns is a list of columns in the referential that we want to 
        return during download
        '''
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        self.upload_config_data(columns, 'link', 'dedupe_linker', config_file_name)
        
    def read_cols_to_return(self, file_role):
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        config = self.read_config_data('link', 'dedupe_linker', config_file_name)
        if not config:
            config = []
        return config
        

    def linker(self, module_name, paths, params):
        '''
        # TODO: This is not optimal. Find way to change paths to smt else
        '''
        
        # Add module-specific paths
        #        if module_name ==  'dedupe_linker':
        #            assert 'train_path' not in paths
        #            assert 'learned_settings_path' not in paths
        #            
        #            paths['train'] = self.path_to('link', module_name, 'training.json')
        #            paths['learned_settings'] = self.path_to('link', module_name, 'learned_settings')
        
        # Initiate log # TODO: move hardcode of file name
        self.mem_data_info['file_role'] = 'link' # Role of file being modified
        self.mem_data_info['file_name'] = 'm3_result.csv' # File being modified
        
        log = self.init_log(module_name, 'link')

        self.mem_data, thresh = MODULES['link'][module_name]['func'](paths, params)
        
        self.mem_data_info['module_name'] = module_name
        
        # Complete log
        log = self.end_log(log, error=False)
                          
        # Update log buffer
        self.log_buffer.append(log)        
        return 


    #==========================================================================
    #  Module specific
    #==========================================================================

    def gen_paths_dedupe(self):        
        self.check_select()
        
        # Get path to training file for dedupe
        training_path = self.path_to('link', 'dedupe_linker', 'training.json')
        learned_settings_path = self.path_to('link', 'dedupe_linker', 'learned_settings')
        
        # Get path to source
        file_name = self.metadata['current']['source']['file_name']
        source_path = self.source.path_to_last_written(module_name=None, 
                    file_name=file_name, before_module='dedupe_linker')
        
        # Get path to ref
        file_name = self.metadata['current']['ref']['file_name']
        ref_path = self.ref.path_to_last_written(module_name=None, 
                    file_name=file_name, before_module='dedupe_linker')
        
        # Add paths
        paths = {
                'ref': ref_path, 
                'source': source_path,
                'train': training_path,
                'learned_settings': learned_settings_path            
                }
        
        return paths

    def select_file(self, file_role, internal, project_id, file_name):
        '''
        Select file to use as source or referential.
        
        INPUT:
            - file_role: "source" or "referential"
            - internal: (bool) is the project available to all (or is it a user project)
            - project_id
            - file_name
        '''
        self.check_file_role(file_role)
        # Check that file exists
        if internal:
            proj = InternalNormalizer(project_id)
        else:
            proj = UserNormalizer(project_id)
            
            
        if file_name not in proj.metadata['files']:
            raise Exception('File {0} could not be found in project {1}'\
                + ' (internal: {2})'.format(file_name, project_id, internal))
        
        # Check that         
        self.metadata['current'][file_role] = {'internal': internal, 
                                             'project_id': project_id,
                                             'file_name': file_name}  
        
        self.write_metadata()

