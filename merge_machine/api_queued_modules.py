#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:10:15 2017

@author: m75380
"""

import os

# Change current path to path of api.py
curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

from normalizer import UserNormalizer
from linker import UserLinker
    
    
def _infer_mvs(project_id, data_params, module_params):
    '''
    Runs the infer_mvs module
    
    wrapper around UserNormalizer.infer ?
    
    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: {
                "module_name": module to fetch from
                "file_name": file to fetch
                }
        - module_params: none
    
    '''
    proj = UserNormalizer(project_id=project_id)
    proj.load_data(data_params['module_name'], data_params['file_name'])    
    result = proj.infer('infer_mvs', module_params)
        
    # Write log
    proj.write_log_buffer(False)
    return result


def _replace_mvs(project_id, data_params, module_params):
    '''
    Runs the mvs replacement module
    
    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: {
                "module_name": module to fetch from
                "file_name": file to fetch
                }
        - module_params: same as result of infer_mvs
    '''
    proj = UserNormalizer(project_id=project_id)

    proj.load_data(data_params['module_name'], data_params['file_name'])
    
    _, run_info = proj.transform('replace_mvs', module_params)
    
    # Write transformations and log
    proj.write_data()
    return run_info

def _infer_types(project_id, data_params, module_params):
    '''
    Runs the infer_types module
    
    wrapper around UserNormalizer.infer ?
    
    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: {
                "module_name": module to fetch from
                "file_name": file to fetch
                }
        - module_params: none
    
    '''
    proj = UserNormalizer(project_id=project_id)
    proj.load_data(data_params['module_name'], data_params['file_name'])    
    result = proj.infer('infer_types', module_params)
        
    # Write log
    proj.write_log_buffer(False)
    return result


def _recode_types(project_id, data_params, module_params):
    '''
    Runs the recoding module
    
    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: {
                "module_name": module to fetch from
                "file_name": file to fetch
                }
        - module_params: same as result of infer_mvs
    '''
    proj = UserNormalizer(project_id=project_id)

    proj.load_data(data_params['module_name'], data_params['file_name'])
    
    _, run_info = proj.transform('recode_types', module_params)
    
    # Write transformations and logs
    proj.write_data()

    return run_info


def _concat_with_init(project_id, data_params, *argv):
    '''
    Concatenate transformed columns with original file 

    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: file to concatenate to original
                {
                    "module_name": module to fetch from
                    "file_name": file to fetch
                }
                
        - module_params: none
    '''
    proj = UserNormalizer(project_id=project_id)

    # TODO: not clean
    if data_params is None:
        (module_name, file_name) = proj.get_last_written()
    else:
        module_name = data_params['module_name']
        file_name = data_params['file_name']

    proj.load_data(module_name, file_name)
    
    # TODO: there was a pdb here. is everything alright ?
    
    _, run_info = proj.transform('concat_with_init', None)
    
    # Write transformations and logs
    proj.write_data()
    return run_info

def _run_all_transforms(project_id, data_params, *argv):
    '''
    Run all transformations that were already (based on presence of 
    run_info.json files) with parameters in run_info.json files.

    ARGUMENTS (GET):
        project_id: ID for "normalize" project

    ARGUMENTS (POST):
        - data_params: file to concatenate to original
                {
                    "file_name": file to use for transform (module_name is 'INIT')
                }
    '''
    proj = UserNormalizer(project_id=project_id)

    file_name = data_params['file_name']

    proj.load_data('INIT', file_name)
    all_run_infos = proj.run_all_transforms()

    # Write transformations and logs
    proj.write_data()
    return all_run_infos


def _create_labeller(project_id, *argv):
    '''
    Create a "dedupe" labeller and pickle to project
    
    ARGUMENTS (GET):
        - project_id
    
    ARGUMENTS (POST):
        - data_params: none
        - module_params: none
    '''
    
    # TODO: data input in gen_dedupe_labeller ?
    proj = UserLinker(project_id=project_id)
    labeller = proj._gen_dedupe_labeller()
    proj.write_labeller(labeller)
    return

# In test_linker
def _linker(project_id, *argv):
    '''
    Runs deduper module. Contrary to other modules, linker modules, take
    paths as input (in addition to module parameters)
    
    ARGUMENTS (GET):
        project_id: ID for "link" project

    ARGUMENTS (POST):
        - data_params: none
        - module_params: none
    '''  
    
    proj = UserLinker(project_id=project_id) # Ref and source are loaded by default
    
    paths = proj._gen_paths_dedupe()
    
    col_matches = proj.read_col_matches()
    my_variable_definition = proj._gen_dedupe_variable_definition(col_matches)
    
    module_params = {
                    'variable_definition': my_variable_definition,
                    'selected_columns_from_source': None,
                    'selected_columns_from_ref': None
                    }  
    
    # TODO: This should probably be moved
    print('Performing deduplication')           
    
    # Perform linking
    proj.linker('dedupe_linker', paths, module_params)

    print('Writing data')
    # Write transformations and log
    proj.write_data()
    
    file_path = proj.path_to(proj.mem_data_info['module_name'], 
                             proj.mem_data_info['file_name'])
    print('Wrote data to: ', file_path)

    return {}

def _test_long(*argv):
    print('-->>>>  STARTED JOB')
    import time
    time.sleep(10)
    print('-->>>>  ENDED JOB')
    return
