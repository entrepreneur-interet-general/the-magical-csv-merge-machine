#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo
"""

import os
import time

from elasticsearch import client, Elasticsearch
from linker import ESLinker
from normalizer import ESNormalizer

from es_connection import es

from CONFIG import LINK_DATA_PATH, NORMALIZE_DATA_PATH

def _check_project_type(project_type):
    if project_type not in ['link', 'normalize']:
        raise Exception('project_type should be link or normalize')

def _check_project_access(project_access):
    if project_access not in ['all', 'public', 'private']:
        raise Exception('project_access should be all, public, or private')


class Admin():
    def __init__(self):
        self.normalize_project_ids = self.list_project_ids('normalize')
        self.link_project_ids = self.list_project_ids('link')
        self.normalize_public_project_ids = self.list_project_ids('normalize', project_access='public')
        self.link_public_project_ids = self.list_project_ids('link', project_access='public')
    
    def path_to(self, project_type, project_id=''):
        '''Returns path to directory of project'''
        _check_project_type(project_type)
        if project_type == 'link':
            data_path = LINK_DATA_PATH
        else:
            data_path = NORMALIZE_DATA_PATH
        return os.path.join(data_path, project_id)

    def list_projects(self, project_type, project_access='all'):
        '''Returns a list of project_metadatas'''
        _check_project_type(project_type)
        _check_project_access(project_access)
        
        list_of_ids = self.list_dirs(project_type)
        list_of_metadatas = []
        
        for id_ in list_of_ids:            
            
            try:
                if project_type == 'link':
                    proj = ESLinker(id_)
                else:
                    proj = ESNormalizer(id_)
                could_load = True
            except:
                could_load = False
                print('Could not load {0}: {1}'.format(project_type, id_))

            if could_load:
                if project_access != 'all':
                    if proj.metadata.get('public', False) == (project_access == 'public'):
                        list_of_metadatas.append(proj.metadata)                
                else:
                    list_of_metadatas.append(proj.metadata)
        return list_of_metadatas

    def list_project_ids(self, project_type, project_access='all'):
        list_of_metadata = self.list_projects(project_type, project_access)
        return {x['project_id'] for x in list_of_metadata}
        

    def list_dirs(self, project_type):
        '''Returns a list of all project_ids'''
        _check_project_type(project_type)
        if os.path.isdir(self.path_to(project_type)):
            return set(filter(lambda x: x[0]!='.', os.listdir(self.path_to(project_type))))
        return {}
    
    def list_projects_by_time(self, project_type, 
                                      project_access='all', 
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
        list_of_metadata = self.list_projects(project_type, project_access)
           
        field = {'created': 'timestamp', 'last_used': 'last_timestamp'}[action]
        mult = {'before': 1, 'after': -1}[when]
        now  = time.time()
        to_return = list(filter(lambda m: mult*(now - m[field]) >= mult*hours_from_now*3600, \
                      list_of_metadata))
        return to_return
        
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
            proj = ESNormalizer(project_id)
        elif project_type == 'link':
            proj = ESLinker(project_id)
            
        proj.delete_project()
        print('Deleted project:', project_type, project_id)
    

# =============================================================================
# Loose link projects (missing normalization projects)
# =============================================================================
    def delete_loose_links(self):
        '''
        Delete link projects for which a normalization project is non existant or not defined
        '''
        for proj_metadata in self.list_projects('link'):
            if (proj_metadata['files']['source'] is None) \
                or (proj_metadata['files']['ref'] is None) \
                or ((proj_metadata['files']['source']['project_id'] not in self.normalize_project_ids)) \
                or ((proj_metadata['files']['ref']['project_id'] not in self.normalize_project_ids)):
                self.remove_project('link', proj_metadata['project_id'])

# =============================================================================
# Elasticsearch
# =============================================================================
    def list_elasticsearch_indices(self):
        ic = client.IndicesClient(es)
        return set(ic.stats()['indices'].keys())
        
    def delete_index(self, index_name):
        ic = client.IndicesClient(es)
        print('Deleting index {0}'.format(index_name))
        return ic.delete(index_name)

    def delete_indices(self, index_names):
        res = []
        for index_name in index_names:
            res.append(self.delete_index(index_name))
        return res
    
    def delete_unused_indices(self, exclude={'123vivalalgerie', '123vivalalgerie2', '123vivalalgerie3', '123vivalalgerie4'}):
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
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up projects based on '
                                     'time of last use or creation')
    
    parser.add_argument('request', 
                        type=str,
                        choices=['list', 'delete'],
                        help='Choose what to do with the results')
    
    parser.add_argument('project_type', 
                        type=str,
                        choices=['link', 'normalize'],
                        help='Type of project to delete')
    
    parser.add_argument('-pa', '--project_access', 
                        type=str,
                        choices=['all', 'public', 'private'],
                        default='private',
                        nargs='?',
                        help='Restrictions on project access')
    
    parser.add_argument('-a', '--action', 
                        type=str, 
                        choices=['created', 'last_used'],
                        default='last_used',
                        nargs='?',
                        help='Whether to filter on date of creation or last use')
    
    parser.add_argument('-w', '--when',
                       type=str,
                       choices=['before', 'after'],
                       default='before',
                       nargs='?',
                       help='Choose whether to delete before or after the target' 
                            'timestamp')
    
    parser.add_argument('-hfn', '--hours_from_now', 
                        type=float,
                        default=24*14, # 14 days
                        nargs='?',
                        help='Number of last hours for which to keep data')
    
    parser.add_argument('-ki', '--keep_indices',
                        action='store_true',
                        help='Flag to keep Elasticsearch indices despite having'
                        ' deleted the MMM projects')    
    parser.add_argument('-kll', '--keep_loose_links',
                        action='store_true',
                        help='Flag to NOT delete loose link projects')    

    args = parser.parse_args()
    
    #args = parser.parse_args('list link -pa all -hfn 24 -w before'.split())
    
    
    
    # Actual requests
    admin = Admin()
    if args.request == 'list':
        func = admin.list_projects_by_time
    elif args.request == 'delete':
        func = admin.remove_project_by_time
    else:
        raise ValueError('No action associated to request: {0}'.format(args.request))
    
    res = func(args.project_type,
         project_access=args.project_access,
         action=args.action, 
         when=args.when, 
         hours_from_now=args.hours_from_now)
    
    print(res)
    
    if (args.request == 'delete'):
        if not args.keep_indices:
            admin.delete_unused_indices()
        if not args.keep_loose_links:
            admin.delete_loose_links()
