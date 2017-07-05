#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:46:00 2017

@author: leo
"""
import gc
import os
import pickle
import time

import pandas as pd

from abstract_data_project import AbstractDataProject
from dedupe_linker import format_for_dedupe, current_load_gazetteer
from labeller import Labeller, DummyLabeller
from normalizer import InternalNormalizer, UserNormalizer

from CONFIG import LINK_DATA_PATH
from MODULES import MODULES, LINK_MODULE_ORDER_log

class Linker(AbstractDataProject):    
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 display_name=None,
                 description=None):
        
        super().__init__(project_id, create_new, display_name=display_name, description=description)
        
        # Add source and ref if the were selected
        if (self.metadata['current']['source'] is not None) \
            and (self.metadata['current']['ref'] is not None):
            self.load_project_to_merge('source')
            self.load_project_to_merge('ref')

    def __repr__(self): 
        string = '{0}({1})'.format(self.__class__.__name__, self.project_id)
        
        string += ' / source: '
        if self.source is not None:
             string += self.source.__repr__()
        else:
            string += 'None'
            
        string += ' / ref: '
        if self.ref is not None:
            string += self.ref.__repr__()
        return string
    
    def __str__(self):
        string = '{0}; project_id:{1}'.format(self.__class__.__name__, self.project_id)
        if self.source is not None:
            string += '\n\n***SOURCE***\n{0}'.format(self.source.__str__())
        if self.ref is not None:
            string += '\n\n***REF***\n{0}'.format(self.ref.__str__())   
        return string
    
    @staticmethod
    def output_file_name(source_file_name):
        '''Name of the file to output'''
        return source_file_name

    def default_log(self):
        '''Default log for a new file'''
        return {module_name: self.default_module_log for module_name in LINK_MODULE_ORDER_log}
    
    def load_project_to_merge(self, file_role):
        '''Uses the "current" field in metadata to load source or ref'''        
        self.check_file_role(file_role)
        # TODO: Add safeguard somewhere
        # Add source
        info = self.metadata['current'][file_role]
        project_id = info['project_id']
        try:
            if info['internal']:
                self.__dict__[file_role] = InternalNormalizer(project_id)
            else:
                self.__dict__[file_role] = UserNormalizer(project_id)
        except:
            self.__dict__[file_role] = None
            #raise Exception('Normalizer project with id {0} could not be found'.format(project_id))
        
    
    @staticmethod
    def check_file_role(file_role):
        if file_role not in ['ref', 'source']:
            raise Exception('file_role should be either "source" or "ref"')

    def check_select(self):
        '''Check that a source and referential were selected'''
        for file_role in ['source', 'ref']:
            if self.metadata['current'][file_role] is None:
                raise Exception('{0} is not defined for this linking project'.format(file_role))

    
    def create_metadata(self, description=None, display_name=None):
        metadata = super().create_metadata()
        metadata['current'] = {'source': None, 'ref': None} # {'source': {internal: False, project_id: "ABC123", file_name: "source.csv.csv"}, 'ref': None}
        return metadata   

    def add_col_matches(self, column_matches):
        '''
        Adds a configuration file with the column matches between source and
        referential.
        column_matches is a json file as dict
        '''
        # TODO: add checks on file
        
        # Add matches
        self.upload_config_data(column_matches, 'dedupe_linker', 'column_matches.json')
        
        # Select these columns for normalization in source and ref
        source_cols = list(set(y for x in column_matches for y in x['source']))
        self.source.add_selected_columns(source_cols)

        ref_cols = list(set(y for x in column_matches for y in x['ref']))
        self.ref.add_selected_columns(ref_cols) 
        
    def read_col_matches(self, add_created=True):
        '''
        Read the column_matches config file and interprets the columns looking
        for processed (normalized) columns
        '''
        config = self.read_config_data('dedupe_linker', 'column_matches.json')
        
        if not config:
            config = []
        #        def expand(cols_to_expand, ref_cols, sep='___'):
        #            new_cols = []
        #            for col in cols_to_expand:
        #                expanded_ver
        #        
        #        # If ___ is found (transformed columns: use that instead)
        #        new_config = []
        #        for match in config:
            
        return config

    def add_col_certain_matches(self, column_matches):
        '''column_matches is a json file as list of dict of list'''
        # TODO: add checks on file
        self.upload_config_data(column_matches, 'dedupe_linker', 'column_certain_matches.json')

    def read_col_certain_matches(self):
        config = self.read_config_data('dedupe_linker', 'column_certain_matches.json')
        if not config:
            config = []
        return config    
    
    def add_cols_to_return(self, file_role, columns):
        '''
        columns is a list of columns in the referential that we want to 
        return during download
        '''
        # Check that both projects are finished
        for file_role in ['source', 'ref']:
            file_name = self.metadata['current'][file_role]['file_name']
            if not self.__dict__[file_role].metadata['complete'][file_name]:
                raise Exception('Cannot select columns: complete {0} project \
                                ({1}) before...'.format(file_role, self.__dict__[file_role].project_id))
        
        # Write columns to return to config
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        self.upload_config_data(columns, 'dedupe_linker', config_file_name)
        
    def read_cols_to_return(self, file_role):
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        config = self.read_config_data('dedupe_linker', config_file_name)
        if not config:
            config = []
        return config


    def add_selected_project(self, file_role, internal, project_id):
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
            
        #        if file_name not in proj.metadata['files']:
        #            raise Exception('File {0} could not be found in project {1} \
        #                 (internal: {2})'.format(file_name, project_id, internal))
        
        # Check that normalization project has only one file (and possibly a MINI__ version)
        if not len(proj.metadata['files']):
            raise Exception('The selected normalization project ({0}) has no upload file'.format(project_id))
        if len(proj.metadata['files']) > 1:
            raise Exception('The selected normalization project ({0}) has more than one file.'\
                    + ' This method expects projects to have exactly 1 file as it'\
                    + ' uses the implicit get_last_written'.format(project_id))
 
        
        (module_name, file_name) = proj.get_last_written()
    
        # TODO: add warning for implicit use of mini
        if proj.metadata['has_mini']:
            proj = 'MINI__' + file_name.replace('MINI__', '')

        # Check that         
        self.metadata['current'][file_role] = {'internal': internal, 
                                             'project_id': project_id,
                                             'module_name': module_name,
                                             'file_name': file_name}
        
        if file_role == 'source':
            self.metadata['log'][self.output_file_name(file_name)] = self.default_log()
        
        self.write_metadata()
        self.load_project_to_merge(file_role)
       
    def read_selected_files(self):
        '''
        Returns self.metadata['current']
        '''
        return self.metadata['current']
    

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
        self.mem_data_info['file_name'] = self.output_file_name(os.path.split(paths['source'])[-1]) # File being modified
        
        log = self.init_active_log(module_name, 'link')

        self.mem_data, run_info = MODULES['link'][module_name]['func'](paths, params)
        
        self.mem_data_info['module_name'] = module_name
        
        # Complete log
        log = self.end_active_log(log, error=False)
                          
        # Update buffers
        self.log_buffer.append(log)        
        self.run_info_buffer[(module_name, self.mem_data_info['file_name'])] = run_info
        return 

    #==========================================================================
    #  Module specific
    #==========================================================================

    def _gen_paths_dedupe(self):        
        self.check_select()
        
        # Get path to training file for dedupe
        training_path = self.path_to('dedupe_linker', 'training.json')
        learned_settings_path = self.path_to('dedupe_linker', 'learned_settings')
        
        # TODO: check that normalization projects are complete ?
        
        # Get path to source
        file_name = self.metadata['current']['source']['file_name']
        source_path = self.source.path_to_last_written(module_name=None, 
                    file_name=file_name)
        
        # Get path to ref
        file_name = self.metadata['current']['ref']['file_name']
        ref_path = self.ref.path_to_last_written(module_name=None, 
                    file_name=file_name)
        
        # Add paths
        paths = {
                'ref': ref_path, 
                'source': source_path,
                'train': training_path,
                'learned_settings': learned_settings_path            
                }
        return paths

    @staticmethod
    def _gen_dedupe_variable_definition(col_matches):
        """Generate my_variable definition for use in dedupe_linker"""
        my_variable_definition = []
        for match in col_matches:
            if (len(match['source']) != 1) or (len(match['ref']) != 1):
                raise Exception('Not dealing with multiple columns (1 source, 1 ref only)')
            my_variable_definition.append({"crf": True, "missing_values": True, "field": 
                {"ref": match['ref'][0], "source": match['source'][0]}, "type": "String"})
        return my_variable_definition

    def _gen_dedupe_dummy_labeller(self):
        '''Return DummyLabeller object'''
        paths = self._gen_paths_dedupe()
        return DummyLabeller(paths, use_previous=True)

    def _gen_dedupe_labeller(self):
        '''Return a Labeller object'''
        # TODO: Add extra config page
        col_matches = self.read_col_matches()
        paths = self._gen_paths_dedupe()
        
        # Generate variable definition for dedupe
        my_variable_definition = self._gen_dedupe_variable_definition(col_matches)
        
        # Put to dedupe input format
        print('loading ref')
        self.load_project_to_merge('ref')
        module_name = self.metadata['current']['ref']['module_name']
        file_name = self.metadata['current']['ref']['file_name']
        self.ref.load_data(module_name, file_name)
        data_ref = format_for_dedupe(self.ref.mem_data, my_variable_definition, 'ref') 
        self.ref.clear_memory()
        gc.collect()
        print('loaded ref')
        
        # Put to dedupe input format
        print('loading source')
        self.load_project_to_merge('source')
        module_name = self.metadata['current']['source']['module_name']
        file_name = self.metadata['current']['source']['file_name']
        self.source.load_data(module_name, file_name)
        data_source = format_for_dedupe(self.source.mem_data, my_variable_definition, 'source') 
        self.source.clear_memory()
        gc.collect()
        print('loaded source')
        
        #==========================================================================
        # Should really start here
        #==========================================================================
        gazetteer = current_load_gazetteer(data_ref, data_source, my_variable_definition)
#                                       og_len_ref=og_len_ref,
#                                       og_len_source=og_len_source)
        
        return Labeller(gazetteer, 
                        training_path=paths['train'], 
                        use_previous=True)

    def write_labeller(self, labeller):
        '''Pickles the labeller object in project'''
        # TODO: Add isinstance(labeller, Labeller)        
        pickle_path = self.path_to('dedupe_linker', 'labeller.pkl')
        with open(pickle_path, 'wb') as w:
            pickle.dump(labeller, w)
    
    def _read_labeller(self):
        '''Reads labeller stored in pickle'''
        pickle_path = self.path_to('dedupe_linker', 'labeller.pkl')
        with open(pickle_path, 'rb') as r:
            labeller  = pickle.load(r)
            
        return labeller

class UserLinker(Linker):
    def path_to(self, module_name='', file_name=''):
        return self._path_to(LINK_DATA_PATH, module_name, file_name)
    

if __name__ == '__main__':
    import json
    import os    
    
    source_file_name = 'source.csv'
    source_user_given_name = 'my_source.csv'
    ref_file_name = 'ref.csv'
    
    # Create source
    proj = UserNormalizer(None, create_new=True)
    source_proj_id = proj.project_id
    
    # Upload files to normalize
    file_path = os.path.join('local_test_data', source_file_name)
    with open(file_path) as f:
        proj.upload_init_data(f, source_file_name, source_user_given_name)

    # Create ref
    proj = UserNormalizer(None, create_new=True)
    ref_proj_id = proj.project_id
    
    # Upload files to normalize
    file_path = os.path.join('local_test_data', ref_file_name)
    with open(file_path) as f:
        proj.upload_init_data(f, ref_file_name, ref_file_name)
    

    # Try deduping
    proj = UserLinker(create_new=True)
    proj.add_selected_project('source', False, source_proj_id)
    proj.add_selected_project('ref', False, ref_proj_id)
    
    paths = dict()
    
    paths = proj._gen_paths_dedupe()
    
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
    with open('local_test_data/integration_1/training.json') as f:
        config_dict = json.load(f)
    proj.upload_config_data(config_dict, 'dedupe_linker', 'training.json')

    # 
    # proj.transform()
    
    # Perform linking
    proj.linker('dedupe_linker', paths, params)
    proj.write_data()   
    
    import pprint
    pprint.pprint(proj.metadata)
       
