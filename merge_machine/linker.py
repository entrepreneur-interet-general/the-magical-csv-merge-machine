#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:46:00 2017

@author: leo
"""
from collections import  defaultdict
import os

from merge_machine.es_labeller import Labeller as ESLabeller
import pandas as pd

from abstract_data_project import ESAbstractDataProject
from normalizer import ESNormalizer
from results_analyzer import link_results_analyzer

from es_connection import es
from CONFIG import LINK_DATA_PATH
from MODULES import LINK_MODULES, LINK_MODULE_ORDER, LINK_MODULE_ORDER_log

class Linker(ESAbstractDataProject):
    MODULES = LINK_MODULES
    MODULE_ORDER = LINK_MODULE_ORDER
    MODULE_ORDER_log = LINK_MODULE_ORDER_log
    
    def __init__(self, 
                 project_id=None, 
                 create_new=False, 
                 display_name=None,
                 description=None,
                 public=False):
        
        super().__init__(project_id, create_new, display_name=display_name, 
                                         description=description, public=public)
        
        # Add source and ref if the were selected
        if (self.metadata['files']['source'] is not None) \
            and (self.metadata['files']['ref'] is not None):
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

    def load_project_to_merge(self, file_role):
        '''Uses the "current" field in metadata to load source or ref'''        
        self._check_file_role(file_role)
        # TODO: Add safeguard somewhere
        # Add source
        
        if file_role == 'source':
            try:
                self.source = ESNormalizer(self.metadata['files']['source']['project_id'])
            except:
                self.source = None
        
        if file_role == 'ref':
            try:
                self.ref = ESNormalizer(self.metadata['files']['ref']['project_id'])
            except:
                self.ref = None            
            #raise Exception('Normalizer project with id {0} could not be found'.format(project_id))
    
    @staticmethod
    def _check_file_role(file_role):
        if file_role not in ['ref', 'source']:
            raise Exception('file_role should be either "source" or "ref"')

    def _check_select(self):
        '''Check that a source and referential were selected'''
        for file_role in ['source', 'ref']:
            if self.metadata['files'][file_role] is None:
                raise Exception('{0} is not defined for this linking project'.format(file_role))
    
    def _create_metadata(self, description=None, display_name=None, public=False):
        metadata = super()._create_metadata(description=description, display_name=display_name, public=public)
        metadata['files'] = {'source': None, 'ref': None} # {'source': {public: False, project_id: "ABC123", file_name: "source.csv.csv"}, 'ref': None}
        metadata['project_type'] = 'link'        
        return metadata   

    def add_col_matches(self, column_matches):
        '''
        Adds a configuration file with the column matches between source and
        referential.
        
        INPUT:
            - column_matches: json file as dict
        '''
        # TODO: add checks on file
        if (self.source is None) or (self.ref is None):
            raise RuntimeError('source or referential were not loaded (add_selected_project) and/or (load_project_to_merge)')
        
        # Add matches
        self.upload_config_data(column_matches, 'es_linker', 'column_matches.json')
        
        # Select these columns for normalization in source and ref
        
        # TODO: this will cover add_certain_col_matches
        # Add to log
        for file_name in self.metadata['log']:
            self.metadata['log'][file_name]['add_selected_columns']['completed'] = True        
        self._write_metadata()   

    def add_es_learned_settings(self, learned_settings):
        '''Adds the learned es configuration'''
        
        print('trying to upload', learned_settings)
        
        # TODO: figure out where to move this
        learned_settings['best_thresh'] = 1
        
        self.upload_config_data(learned_settings, 'es_linker', 'learned_settings.json')
        
        for file_name in self.metadata['log']:
            self.metadata['log'][file_name]['upload_es_train']['completed'] = True   
        self._write_metadata()
        
    def read_col_matches(self, add_created=True):
        '''
        Read the column_matches config file and interprets the columns looking
        for processed (normalized) columns
        '''
        config = self.read_config_data('es_linker', 'column_matches.json')
        
        if not config:
            config = []
            
        return config

    def add_col_certain_matches(self, column_matches):
        '''column_matches is a json file as list of dict of list'''
        # TODO: add checks on file
        self.upload_config_data(column_matches, 'es_linker', 'column_certain_matches.json')

    def read_col_certain_matches(self):
        config = self.read_config_data('es_linker', 'column_certain_matches.json')
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
            file_name = self.metadata['files'][file_role]['file_name']
            if not self.__dict__[file_role].metadata['complete'][file_name]:
                raise Exception('Cannot select columns: complete {0} project \
                                ({1}) before...'.format(file_role, self.__dict__[file_role].project_id))
        
        # Write columns to return to config
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        self.upload_config_data(columns, 'es_linker', config_file_name)
        
    def read_cols_to_return(self, file_role):
        config_file_name = 'columns_to_return_{0}.json'.format(file_role)
        config = self.read_config_data('es_linker', config_file_name)
        if not config:
            config = []
        return config


    def add_selected_project(self, file_role, public, project_id):
        '''
        Select file to use as source or referential.
        
        INPUT:
            - file_role: "source" or "referential"
            - public: (bool) is the project available to all (or is it a user project)
            - project_id
            - file_name
        '''
        self._check_file_role(file_role)
        # Check that file exists
        if public:
            raise DeprecationWarning
        else:
            proj = ESNormalizer(project_id)
            
        #        if file_name not in proj.metadata['files']:
        #            raise Exception('File {0} could not be found in project {1} \
        #                 (public: {2})'.format(file_name, project_id, public))
        
        # Check that normalization project has only one file (and possibly a MINI__ version)
        if not len(proj.metadata['files']):
            raise Exception('The selected normalization project ({0}) has no upload file'.format(project_id))
        if len(proj.metadata['files']) > 1:
            raise Exception('The selected normalization project ({0}) has more than one file.'\
                    + ' This method expects projects to have exactly 1 file as it'\
                    + ' uses the implicit get_last_written'.format(project_id))
 
        # TODO: last written is a bad idea because if we modify normalization then BOOM !
        # TODO: last_written btw concat_with_initi and init ?
        (module_name, file_name) = proj.get_last_written()
    
        # TODO: add warning for implicit use of not-MINI
        if proj.metadata['has_mini'] and (file_role == 'source'):
            file_name = file_name.replace('MINI__', '')
        if proj.metadata['has_mini'] and (file_role == 'ref'):
            file_name = file_name.replace('MINI__', '')

        # Check that         
        self.metadata['files'][file_role] = {'public': public, 
                                             'project_id': project_id,
                                             'module_name': module_name,
                                             'file_name': file_name,
                                             'restricted': False}
        
        # Create log for source
        if file_role == 'source':
            self.metadata['log'][self.output_file_name(file_name)] = self._default_log()
        
        # Add project selection 
        if (self.metadata['files']['source'] is not None) and (self.metadata['files']['ref'] is not None):
            for file_name in self.metadata['log']:
                self.metadata['log'][file_name]['INIT']['completed'] = True
        self._write_metadata()
        self.load_project_to_merge(file_role)
       
    def read_selected_files(self):
        '''
        Returns self.metadata['files']
        '''
        return self.metadata['files']
    
    def infer(self, module_name, params):
        '''Overwrite to allow restrict_reference'''
        if module_name == 'infer_restriction':
            params['NO_MEM_DATA'] = True
        return super().infer(module_name, params)
    
    def linker(self, module_name, data_params, module_params):
        '''Wrapper around link methods.'''
        
        if module_name == 'es_linker':
            return self.es_linker(module_params)
        elif module_name == 'dedupe_linker':
            raise DeprecationWarning

    def es_linker(self, module_params):
        module_params['index_name'] = ESNormalizer(self.ref.project_id).index_name
        
        s = self.metadata['files']['source']
        
        self.source.load_data(s['module_name'], s['file_name'])
        
        self.mem_data = self.source.mem_data
        self.mem_data_info = self.source.mem_data_info
        
        # Change file_name to output file_name
        self.mem_data_info['file_name'] = self.output_file_name(self.mem_data_info['file_name']) # File being modified

        log, run_info = self.transform('es_linker', module_params)        
        
        return log, run_info

    #==========================================================================
    #  Module specific: ES Linker
    #==========================================================================

    def _gen_paths_es(self):        
        self._check_select()
        
        # Get path to training file for ES linker
        training_path = self.path_to('es_linker', 'training.json')
        learned_settings_path = self.path_to('es_linker', 'learned_settings.json')
        
        # TODO: check that normalization projects are complete ?
        
        # Get path to source
        # TODO: fix this: use current
        file_name = self.metadata['files']['source']['file_name']
        source_path = self.source.path_to_last_written(module_name=None, 
                    file_name=file_name)
        
        # Add paths
        paths = {
                'source': source_path,
                'train': training_path,
                'learned_settings': learned_settings_path            
                }
        return paths

    @staticmethod
    def _tuple_or_string(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, list):
            if len(x) == 1:
                return x[0]
            else:
                return tuple(x)
        elif isinstance(x, tuple):
            if len(x) == 1:
                return x[0]
            else:
                return x
        else:
            raise ValueError('Value should be str, list or tuple')

    def _gen_es_labeller(self, columns_to_index=None, certain_column_matches=None):
        '''
        Return a es_labeller object
        '''
        self._check_select()
        
        #chunksize = 40000
        
        col_matches_tmp = self.read_col_matches()
        col_matches = []
        for match in col_matches_tmp:
            col_matches.append({'source': self._tuple_or_string(match['source']), 
                                'ref': self._tuple_or_string(match['ref'])})
        # TODO: lists to tuple in col_matches
        
        paths = self._gen_paths_es()
        source = pd.read_csv(paths['source'], 
                            sep=',', encoding='utf-8',
                            dtype=str, nrows=3000)
        source = source.where(source.notnull(), '')
        
        ref_table_name = self.ref.project_id
        if columns_to_index is None:
            columns_to_index = self.ref.gen_default_columns_to_index()
        
        labeller = ESLabeller(es, source, ref_table_name, col_matches, columns_to_index, certain_column_matches)
        
        # TODO: Auto label certain pairs 
        
        # TODO: Add pre-load for 3 first queries
    
        return labeller
    
    def labeller_to_json(self, labeller):
        
        file_path = self.path_to('es_linker', 'labeller.json')
        labeller.to_json(file_path)
        
    def labeller_from_json(self):
        file_path = self.path_to('es_linker', 'labeller.json')
        
        paths = self._gen_paths_es()
        source = pd.read_csv(paths['source'], 
                            sep=',', encoding='utf-8',
                            dtype=str, nrows=3000)
        source = source.where(source.notnull(), '')        
        
        
        ref_table_name = self.ref.project_id
        labeller = ESLabeller.from_json(file_path, es, source, ref_table_name)
        
        return labeller
            

    def analyze_results(self, params={}):
        # Check that memory is loaded (if necessary)
        self._check_mem_data()
        
        module_name = 'link_results_analyzer'
        
        # Initiate log
        log = self._init_active_log(module_name, 'infer')
        
        complete_metrics = defaultdict(int) 
        for data in self.mem_data:
            metrics = link_results_analyzer(data, params)
            
            for col in ['num_match_thresh', 'num_match', 'num_verif_samples']:
                complete_metrics[col] += metrics[col]
            
            # Weigh ratios according to the number of samples (we divide after)
            complete_metrics['perc_match_thresh'] += metrics['perc_match_thresh'] * metrics['num_match_thresh']
            complete_metrics['perc_match'] += metrics['perc_match'] * metrics['num_match']
            complete_metrics['precision'] += metrics.get('precision', 0) * metrics['num_verif_samples']
            
        if complete_metrics['num_match_thresh']:
            complete_metrics['perc_match_thresh'] /= complete_metrics['num_match_thresh']

        if complete_metrics['num_match']:
            complete_metrics['perc_match'] /= complete_metrics['num_match']
            
        if complete_metrics['precision']:
            complete_metrics['precision'] /= complete_metrics['num_verif_samples']            

        # Write result of inference
        module_to_write_to = self.MODULES['infer'][module_name]['write_to']

        self.upload_config_data(complete_metrics, module_to_write_to, 'infered_config.json')
        
        # Update log buffer
        self._end_active_log(log, error=False)  
        
        return complete_metrics    

#    def create_es_index_ref(self, columns_to_index, force=False):
#        '''#TODO: doc'''
#        
#        self.ref = ESNormalizer(self.ref.project_id)
#        
#        # TODO: Doesn't seem safe..
#        (module_name, file_name) = proj.get_last_written(file_name=self.metadata['files']['ref']['file_name'])
#        ref_path = self.ref.path_to(module_name,file_name)
#        return self.ref.create_index(ref_path, columns_to_index, force)


    #==========================================================================
    #  Module specific: Restriction
    #==========================================================================

#    training_df = training_to_ref_df(training)
#    common_words = find_common_words(training_df)
#    common_vals = find_common_vals(training_df)    
    
    #    def perform_restriction(self, params):
    #        '''
    #        Writes a new file with the path restricted reference
    #        
    #        /!\ Contrary to infer or transform, the log is written directly.
    #        '''
    #        
    #        current_module_name = 'restriction'
    #        
    #        # Initiate log
    #        self.mem_data_info['file_role'] = 'link' # Role of file being modified
    #        
    #        log = self._init_active_log(current_module_name, 'link')
    #        
    #        # TODO: Move this
    #        self.load_project_to_merge('ref')
    #        module_name = self.metadata['files']['ref']['module_name']
    #        file_name = self.metadata['files']['ref']['file_name']
    #        
    #        self.ref.load_data(module_name, file_name, restrict_to_selected=False)   
    #        
    #        self.mem_data = (perform_restriction(part_tab, params)[0] \
    #                                   for part_tab in self.ref.mem_data) # TODO: no run info !
    #        
    #        # Complete log
    #        self.log_buffer.append(self._end_active_log(log, error=False))    
    #        self.mem_data_info['file_name'] = self.ref.mem_data_info['file_name']
    #        self.mem_data_info['module_name'] = current_module_name        
    #        
    #        # TODO: fix fishy:
    #        #        self.run_info_buffer[(current_module_name, '__REF__')] = {}
    #        #        self.run_info_buffer[(current_module_name, '__REF__')][current_module_name] = run_info # TODO: fishy
    #        
    #        # Add restricted to current for restricted
    #        self.metadata['files']['ref']['restricted'] = True
    #        
    #        # TODO: write new_ref to "restriction"
    #        self.write_data()
    #        self.clear_memory()
    #        
    #        return {} #run_info

        # TODO: Add to current reference
        
        # TODO: Return smth

#    
#
#        
#        
#        self.mem_data_info['file_role'] = 'link' # Role of file being modified
#        self.mem_data_info['file_name'] = self.output_file_name(os.path.split(paths['source'])[-1]) # File being modified
#        
#        log = self._init_active_log(module_name, 'link')
#
#        self.mem_data, run_info = MODULES['link'][module_name]['func'](paths, params)
#        
#        self.mem_data_info['module_name'] = module_name
#        
#        # Complete log
#        log = self._end_active_log(log, error=False)
#                          
#        # Update buffers
#        self.log_buffer.append(log)        
#        self.run_info_buffer[(module_name, self.mem_data_info['file_name'])] = run_info
#        return 

class ESLinker(Linker):
    def path_to(self, module_name='', file_name=''):
        return self._path_to(LINK_DATA_PATH, module_name, file_name)
    

if __name__ == '__main__':
    
    source_file_name = 'source.csv'
    source_user_given_name = 'my_source.csv'
    ref_file_name = 'ref.csv'
    
    # Create source
    proj = ESNormalizer(None, create_new=True)
    source_proj_id = proj.project_id
    
    # Upload files to normalize
    file_path = os.path.join('local_test_data', source_file_name)
    with open(file_path, 'rb') as f:
        proj.upload_init_data(f, source_file_name, source_user_given_name)

    # Create ref
    proj = ESNormalizer(None, create_new=True)
    ref_proj_id = proj.project_id
    
    # Upload files to normalize
    file_path = os.path.join('local_test_data', ref_file_name)
    with open(file_path, 'rb') as f:
        proj.upload_init_data(f, ref_file_name, ref_file_name)
    

    # Try deduping
    proj = ESLinker(create_new=True)
    
    proj.add_selected_project('source', False, source_proj_id)
    proj.add_selected_project('ref', False, ref_proj_id)
    

    # Index
    proj.load_project_to_merge('ref')

    ref = ESNormalizer(proj.ref.project_id)
    
    # ref_path, columns_to_index, force=False)
    ref_path = ref.path_to_last_written()
    
    columns_to_index = {
        'numero_uai': {},
        'denomination_principale_uai': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'patronyme_uai': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'adresse_uai': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'localite_acheminement_uai': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'departement': {
            'french', 'whitespace', 'integers', 'n_grams'
        },
        'code_postal_uai': {},
        'full_name': {
            'french', 'whitespace', 'integers', 'n_grams'
        }
    }
    
    ref.create_index(ref_path, columns_to_index, force=False)
    
    # Link
    index_name = proj.metadata['files']['ref']['project_id']
    query_template = (('must', 'commune', 'localite_acheminement_uai', '.french', 1), ('must', 'lycees_sources', 'full_name', '.french', 1))
    threshold = 3.5
    must = {'full_name': ['lycee']}
    must_not = {'full_name': ['ass', 'association', 'sportive', 'foyer']}

    params=dict()
    params['index_name'] = index_name
    params['query_template'] = query_template
    params['thresh'] = threshold
    params['must'] = must
    params['must_not'] = must_not
    
    proj.linker('es_linker', None, params)

    proj.write_data()   

    
    import pprint
    pprint.pprint(proj.metadata)
       
