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
        if (file_role is not None) and (file_role != 'referentials'):
            raise Exception('"file_role" should be "referentials" for internal \
                            referentials')
    
    def path_to(self, file_role='', module_name='', file_name=''):
        '''
        Return path to directory that stores specific information for a project 
        module
        '''
        if file_role:
            self.check_file_role(file_role)
        path = os.path.join(DATA_PATH, file_role, module_name, file_name)
        return os.path.abspath(path)    

    def create_metadata(self):
        metadata = dict()
        metadata['timestamp'] = time.time()
        metadata['description'] = ''
        metadata['log'] = []
        return metadata  
    
    def add_description(self, description):
        '''Add description to metadata'''
        self.metadata['description'] = description
