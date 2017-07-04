#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo
"""

import os
import shutil

from linker import UserLinker
from normalizer import UserNormalizer

from CONFIG import LINK_DATA_PATH, NORMALIZE_DATA_PATH

def _check_project_type(project_type):
    if project_type not in ['link', 'normalize']:
        raise Exception('project_type should be link or normalize')

class Admin():
    def __init__(self):
        self.normalize_project_ids = self.list_projects('normalize')
        self.link_project_ids = self.list_projects('normalize')
        self.internal_project_ids = [] # TODO: implement this
    
    def path_to(self, project_type, project_id=''):
        '''Returns path to directory of project'''
        _check_project_type(project_type)
        if project_type == 'link':
            data_path = LINK_DATA_PATH
        else:
            data_path = NORMALIZE_DATA_PATH
        return os.path.join(data_path, project_id)
    
    def list_project_ids(self, project_type):
        '''Returns a list of all project_ids'''
        _check_project_type(project_type)
        if os.path.isdir(self.path_to(project_type)):
            return os.listdir(self.path_to(project_type))
        return []
    
    def list_projects(self, project_type):
        '''Returns a list of project_metadatas'''
        list_of_ids = self.list_project_ids(project_type)
        
        list_of_metadatas = []
        for _id in list_of_ids:
            if project_type == 'link':
                proj = UserLinker(_id)
            else:
                proj = UserNormalizer(_id)
            list_of_metadatas.append(proj.metadata)
        return list_of_metadatas
    
    def remove_project(self, project_type, project_id):
        assert project_id and (project_id is not None)
        _check_project_type(project_type)
        dir_path = self.path_to(project_type, project_id) 
        if not os.path.isdir(dir_path):
            raise Exception('No project found with the following ID: {0}'.format(project_id))
        shutil.rmtree(dir_path)
    
        
if __name__ == '__main__':
    admin = Admin()
    
    id_ = '8b3fbab040534afae137e2ae124ce152'
    
    for id_ in admin.list_projects():
        proj = UserNormalizer(id_)
        print(proj.project_id, proj.time_since_last_action())
        print(proj.list_csvs())
            
    
