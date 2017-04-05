#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:14:09 2017

@author: leo

# TODO:
    - Automate acquisition
    - Have "file info" with source/ last update, link to transformation code...


"""
import os
import time

from project import Project

from CONFIG import DATA_PATH

class Referential(Project):
    """
    This class provides tools to manage internal referentials.
    """
    def check_file_role(self, file_role):
        if (file_role is not None) and (file_role != 'ref'):
            raise Exception('"file_role" should be "ref" for internal \
                            referentials (input: {0})'.format(file_role))
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        if file_role:
            self.check_file_role(file_role)
        path = os.path.join(DATA_PATH, 'referentials', self.project_id, 
                            file_role, module_name, file_name)
        return os.path.abspath(path)    

    def create_metadata(self, description=''):
        metadata = dict()
        metadata['timestamp'] = time.time()
        metadata['description'] = description
        metadata['current'] = {'source': None, 'ref': None} 
        metadata['log'] = []
        metadata['project_id'] = self.project_id
        return metadata  
    
    def add_description(self, description):
        '''Add description to metadata'''
        self.metadata['description'] = description



    def select_file(self, file_role, file_name, internal=False, project_id=None):
        # TODO: was initially in user_project
        # TODO: Validate that the file exists
        self.check_file_role(file_role)
        self.metadata['current'][file_role] = {'internal': internal, 'file_name': file_name}  
        
        if internal:
            raise Exception('Trying to use internal referential from within\
                            an internal referential')
        if project_id is not None:
            self.metadata['current'][file_role]['project_id'] = project_id
        else:
            assert internal
            self.metadata['current'][file_role]['project_id'] = self.project_id
        self.write_metadata()