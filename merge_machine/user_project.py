#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:04:51 2017

@author: leo
"""
import os
import time

from project import Project

from CONFIG import DATA_PATH


class UserProject(Project):
    """
    This class provides tools to manage user projects
    """
    def check_file_role(self, file_role):
        if (file_role not in ['ref', 'source']) and (file_role is not None):
            raise Exception('"file_role" is either "source" or "ref"')
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        if file_role:
            self.check_file_role(file_role)
        path = os.path.join(DATA_PATH, 'projects', self.project_id, file_role, module_name, file_name)
        return os.path.abspath(path)    

    def create_metadata(self):
        metadata = dict()
        metadata['timestamp'] = time.time()
        metadata['user_id'] = 'NOT IMPlEMENTED'
        metadata['use_internal_ref'] = None
        metadata['internal_ref_name'] = None
        metadata['source_names'] = []
        metadata['log'] = []
        metadata['project_id'] = self.project_id
        return metadata   
    

if __name__ == '__main__':
    # Create/Load a project
    project_id = "f87cf0519b713abd8f40cdd11d564f98"
    proj = UserProject(None)
    
    # Upload source to project
    file_name = 'source.csv'
    file_path = os.path.join('local_test_data', file_name)
    with open(file_path) as f:
        proj.add_init_data(f, 'source', file_name)
        
        
    
    # Load source data to memory
    proj.load_data(file_role='source', module_name='INIT' , file_name=file_name)
    
    infered_params = proj.infer('infer_mvs', None)
    
    # Try transformation
    params = {'mvs_dict': {'all': [],
              'columns': [{'col_name': u'uai',
                           'missing_vals': [{'origin': ['len_ratio'],
                                             'score': 0.2,
                                             'val': u'NR'}]}]},
                'thresh': 0.6}
    log = proj.transform('replace_mvs', params)
    
    # Write transformed file
    proj.write_data()
    proj.write_log_buffer(written=True)
    
    # Remove previously uploaded file
    # proj.remove_data('source', 'INIT', 'source.csv')    
    import pprint
    pprint.pprint(log)
    pprint.pprint(proj.get_arb())
    pprint.pprint(proj.metadata)
    
    proj.clean_metadata('source', 'source.csv') 