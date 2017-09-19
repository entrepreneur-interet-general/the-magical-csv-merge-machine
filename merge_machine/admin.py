#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo
"""

import os
import time

from elasticsearch import client, Elasticsearch
from linker import UserLinker
from normalizer import ESReferential

from CONFIG import LINK_DATA_PATH, NORMALIZE_DATA_PATH

def _check_project_type(project_type):
    if project_type not in ['link', 'normalize']:
        raise Exception('project_type should be link or normalize')

class Admin():
    def __init__(self):
        self.normalize_project_ids = self.list_projects('normalize')
        self.link_project_ids = self.list_projects('normalize')
        self.public_project_ids = [] # TODO: implement this
    
    def path_to(self, project_type, project_id=''):
        '''Returns path to directory of project'''
        _check_project_type(project_type)
        if project_type == 'link':
            data_path = LINK_DATA_PATH
        else:
            data_path = NORMALIZE_DATA_PATH
        return os.path.join(data_path, project_id)

    def list_projects(self, project_type, public_only=False):
        '''Returns a list of project_metadatas'''
        _check_project_type(project_type)
        
        list_of_ids = self.list_dirs(project_type)
        list_of_metadatas = []
        for id_ in list_of_ids:
            if project_type == 'link':
                try:
                    proj = UserLinker(id_)
                except:
                    print('here', id_)
            else:
                proj = ESReferential(id_)
            if proj.metadata.get('public', False):
                list_of_metadatas.append(proj.metadata)
        return list_of_metadatas

    def list_project_ids(self, project_type, public_only=False):
        list_of_metadata = self.list_projects(project_type, public_only)
        return {x['project_id'] for x in list_of_metadata}
        

    def list_dirs(self, project_type):
        '''Returns a list of all project_ids'''
        _check_project_type(project_type)
        if os.path.isdir(self.path_to(project_type)):
            return set(filter(lambda x: x[0]!='.', os.listdir(self.path_to(project_type))))
        return {}
    
    def list_projects_by_time(self, project_type, 
                                      public_only=False, 
                                      action='created', 
                                      when='before', 
                                      hours_from_now=24*7):
        '''
        INPUT:
            project_type: normalize or link
            action: 'created' or 'last_used'
            when: 'before' or 'after' (both are inclusive, but proba of equality is very slim)
            hours_from_now: how many hours before current time are we looking at
        '''
        list_of_metadata = self.list_projects(project_type, public_only)
           
        field = {'created': 'timestamp', 'last_used': 'last_timestamp'}[action]
        mult = {'before': 1, 'after': -1}[when]
        now  = time.time()
        return filter(lambda m: mult*(now - m[field]) >= mult*hours_from_now*3600, \
                      list_of_metadata)
        
    def remove_project_by_time(self, project_type, **kwargs):
        '''Same keywords as list_projects_by_time'''
        for project in self.list_projects_by_time(project_type, **kwargs):
            self.remove_project(project_type, project['project_id'])

    def remove_project(self, project_type, project_id):
        assert project_id and (project_id is not None)
        _check_project_type(project_type)
        dir_path = self.path_to(project_type, project_id) 
        if not os.path.isdir(dir_path):
            raise Exception('No project found with the following ID: {0}'.format(project_id))
            
        if project_type == 'normalize':
            proj = ESReferential(project_id)
        elif project_type == 'link':
            proj = UserLinker(project_id)
            
        proj.delete_project()
        print('Deleted project:', project_type, project_id)
    

# =============================================================================
# Elasticsearch
# =============================================================================
    def list_elasticsearch_indices(self):
        es = Elasticsearch()
        ic = client.IndicesClient(es)
        return set(ic.stats()['indices'].keys())
        
    def delete_index(self, index_name):
        es = Elasticsearch()
        ic = client.IndicesClient(es)
        return ic.delete(index_name)

    def delete_indices(self, index_names):
        res = []
        for index_name in index_names:
            res.append(self.delete_index(index_name))
        return res
    
    def delete_unused_indices(self, exclude={'123vivalalgerie', '123vivalalgerie3'}):
        '''
        Delete ES indices whose name are not present among the normalisation
        projects names.
        '''
        indices_to_delete =  self.list_elasticsearch_indices() \
                            - self.list_project_ids('normalize') \
                            - exclude
        self.delete_indices(indices_to_delete)
    
# =============================================================================
# 
# =============================================================================
    
if __name__ == '__main__':
    admin = Admin()
    admin.remove_project_by_time('link', 
                                 action='created', 
                                 when='before', 
                                 hours_from_now=24*30)
            
    
