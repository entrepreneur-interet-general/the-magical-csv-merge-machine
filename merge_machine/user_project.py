#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:04:51 2017

@author: leo


TODO: Add private flag to user project so you have to log in to see/use

TODO: rename column header on upload (remove whitespaces)

TODO: Categorial extraction

"""
import os
import time

from project import MODULES, Project
from referential import Referential

from CONFIG import DATA_PATH

class UserProject(Project):
    """
    This class provides tools to manage user projects
    """
    def __init__(self, project_id=None, create_new=False, description=''):
        if (project_id is not None) and create_new:
            raise Exception('Set project_id to None or create_new to False')
        super().__init__(project_id, create_new, description)
    
    def check_file_role(self, file_role):
        if (file_role not in ['ref', 'source', 'link']) and (file_role is not None):
            raise Exception('"file_role" is either "source" or "ref" or "link"')
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        if file_role is None:
            file_role = ''
        if module_name is None:
            module_name = ''
        if file_name is None:
            file_name = ''
            
        if file_role:
            self.check_file_role(file_role)
        
        path = os.path.join(DATA_PATH, 'projects', self.project_id, file_role, 
                                module_name, file_name)

        return os.path.abspath(path)    

    def create_metadata(self, description=''):
        metadata = dict()
        metadata['timestamp'] = time.time()
        metadata['user_id'] = 'NOT IMPlEMENTED'
        metadata['use_internal_ref'] = None
        metadata['ref_name'] = None
        metadata['current'] = {'source': None, 'ref': None} # {'source': {internal: False, file_name: "source.csv.csv"}, 'ref': None}
        metadata['description'] = description
        metadata['log'] = []
        metadata['project_id'] = self.project_id
        return metadata   

    def select_file(self, file_role, file_name, internal=False):
        # TODO: Validate that the file exists
        self.check_file_role(file_role)
        self.metadata['current'][file_role] = {'internal': internal, 'file_name': file_name}  
        self.write_metadata()


    def add_col_matches(self, column_matches):
        '''column_matches is a json file as list of dict of list'''
        # TODO: add checks on file
        self.upload_config_data(column_matches, 'link', 'dedupe_linker', 'column_matches.json')

    def read_col_matches(self):
        return self.read_config_data('link', 'dedupe_linker', 'column_matches.json')


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
        # Get path to training file for dedupe
        training_path = self.path_to('link', 'dedupe_linker', 'training.json')
        learned_settings_path = self.path_to('link', 'dedupe_linker', 'learned_settings')
        
        # Get path to source
        if self.metadata['current']['source']['internal']:
            raise Exception('Not dealing with internal sources yet')
        else:
            (file_role, module_name, file_name) = self.get_last_written('source', 
                                None, self.metadata['current']['source']['file_name'])
        source_path = self.path_to(file_role, module_name, file_name)
        
        # Get path to ref
        if self.metadata['current']['ref']['internal']:
            raise Exception('Not dealing with internal sources yet')
        else:
            (file_role, module_name, file_name) = self.get_last_written('ref', 
                                None, self.metadata['current']['ref']['file_name'])
        ref_path = self.path_to(file_role, module_name, file_name)
    
        # Add paths
        paths = {
                'ref': ref_path, 
                'source': source_path,
                'train': training_path,
                'learned_settings': learned_settings_path            
                }
        
        return paths

if __name__ == '__main__':
    import json
    
    # Create/Load a project
    project_id = "4e8286f034eef40e89dd99ebe6d87f21"
    proj = UserProject(None, create_new=True)
    
    # Upload source to project
    file_names = ['source.csv']
    for file_name in file_names:
        file_path = os.path.join('local_test_data', file_name)
        with open(file_path) as f:
            proj.add_init_data(f, 'source', file_name)

    # Upload ref to project
    file_path = 'local_test_data/ref.csv'
    with open(file_path) as f:
        proj.add_init_data(f, 'ref', file_name)

    # Load source data to memory
    proj.load_data(file_role='source', module_name='INIT' , file_name='source.csv')
    
    
    infered_params = proj.infer('infer_mvs', params=None)
    
    # Try transformation
    params = {
                'mvs_dict': {'all': [],
                           'columns': {'uai': 
                               [{'origin': ['len_ratio'], 'score': 0.2,'val': u'NR'}]
                                    },
                            },
                'thresh': 0.6
            }
    proj.transform('replace_mvs', params)
    
    # Write transformed file
    proj.write_data()
    proj.write_log_buffer(written=True)
    
    # Remove previously uploaded file
    # proj.remove_data('source', 'INIT', 'source.csv')    

    # Try deduping
    paths = dict()
    
    (file_role, module_name, file_name) = proj.get_last_written(file_role='ref')
    paths['ref'] = proj.path_to(file_role, module_name, file_name)
    
    (file_role, module_name, file_name) = proj.get_last_written(file_role='source')
    paths['source'] = proj.path_to(file_role, module_name, file_name)
    
    
    paths['train'] = proj.path_to('link', 'dedupe_linker', 'training.json')
    paths['learned_settings'] = proj.path_to('link', 'dedupe_linker', 'learned_settings')
    
    ## Parameters
    # Variables
    my_variable_definition = [
                            {'field': 
                                    {'source': 'lycees_sources',
                                    'ref': 'full_name'}, 
                            'type': 'String', 
                            'crf':True, 
                            'missing_values':True},
                            
                            {'field': {'source': 'commune', 
                                       'ref': 'localite_acheminement_uai'}, 
                            'type': 'String', 
                            'crf': True, 
                            'missing_values':True}
                            ]

    # What columns in reference to include in output
    selected_columns_from_ref = ['numero_uai', 'patronyme_uai', 
                                 'localite_acheminement_uai']
    
    #                          
    params = {'variable_definition': my_variable_definition,
              'selected_columns_from_ref': selected_columns_from_ref}

    # Add training data
    with open('local_test_data/training.json') as f:
        config_dict = json.load(f)
    proj.upload_config_data(config_dict, 'link', 'dedupe_linker', 'training.json')
                
              
              
    proj.linker('dedupe_linker', paths, params)
    proj.write_data()
    proj.write_log_buffer(written=True)
    
    
    
    import pprint
    pprint.pprint(proj.get_arb())
    pprint.pprint(proj.metadata)
    
    pprint.pprint(proj.log_by_file_name())
        