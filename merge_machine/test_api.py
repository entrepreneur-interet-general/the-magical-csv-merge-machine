#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:18 2017

@author: leo


TODO:
    - Use external json as parameters for testing

"""

import json
import pprint
import requests
import time

PROTOCOL = 'http://'
HOST = '0.0.0.0:5000'
PRINT = True
PRINT_CHAR_LIMIT = 10000

def my_pformat(dict_obj):
    formated_string = pprint.pformat(dict_obj)
    if len(formated_string) > PRINT_CHAR_LIMIT:
        formated_string = formated_string[:PRINT_CHAR_LIMIT]
        formated_string += '\n[ ... ] (increase PRINT_CHAR_LIMIT to see more...)'
    return formated_string

def my_print(func):
    def wrapper(*args, **kwargs):
        if PRINT:
            print('\n' + '>'*60 + '\n', args[0])
            
            if len(args) >= 2:
                body = args[1]
                if body:
                    print('\n <> POST REQUEST:\n', my_pformat(body))
        resp = func(*args, **kwargs)
        
        if PRINT:
            print('\n <> RESPONSE:\n', my_pformat(resp))        
        
        return resp
    return wrapper
    
@my_print
def get_resp(url_to_append):
    url = PROTOCOL + HOST + url_to_append
    resp = requests.get(url)
    
    if resp.ok:
        parsed_resp = json.loads(resp.content.decode())
        #        _print(url_to_append,  parsed_resp)
        return parsed_resp
    else: 
        raise Exception('Problem:\n', resp)

@my_print
def post_resp(url_to_append, body, **kwargs):
    url = PROTOCOL + HOST + url_to_append
    resp = requests.post(url, json=body, **kwargs)     
    
    if resp.ok:
        parsed_resp = json.loads(resp.content.decode())
        #        _print(url_to_append, parsed_resp)
        return parsed_resp
    else: 
        raise Exception('Problem:\n', resp)

def wait_get_resp(url_to_append, max_wait=10):
    url = PROTOCOL + HOST + url_to_append
    start_time = time.time()
    while (time.time() - start_time) <= max_wait:
        resp = requests.get(url)
        if resp.ok:
            parsed_resp = json.loads(resp.content.decode())
        else: 
            raise Exception('Problem:\n', resp)
            
        if parsed_resp['completed']:
            if PRINT:
                print('\n <> RESPONSE AFTER JOB COMPLETION (Waited {0} seconds):'.format(time.time()-start_time))
                print(parsed_resp)
            return parsed_resp
        time.sleep(0.25)
    print(time.time() - start_time)
    raise Exception('Timed out after {0} seconds'.format(max_wait))



def normalize_pipeline(params):
    #==============================================================================
    # Create new normalization project
    #==============================================================================
    url_to_append = '/api/new/normalize'
    #    body = {'description': 'File to use as source for testing',
    #            'display_name': 'test_source.csv',
    #            'internal': False}
    body = params['new_project']
    resp = post_resp(url_to_append, body)
    project_id = resp['project_id']
    
    #==============================================================================
    # Upload new file
    #==============================================================================
    url_to_append = '/api/normalize/upload/{0}'.format(project_id)
    # file_path = 'local_test_data/integration_1/source.csv'
    # file_name = file_path.rsplit('/', 1)[-1]
    file_path = params['file_path']
    with open(file_path, 'rb') as f:
        resp = post_resp(url_to_append, 
                         body, 
                         files={'file': f})
    
    #==============================================================================
    # Get last written
    #==============================================================================
    url_to_append = '/api/last_written/normalize/{0}'.format(project_id)
    body = {
            # 'module_name': 
            #'file_name':
            'before_module': 'replace_mvs'        
            }
    resp = post_resp(url_to_append, body)
    module_name = resp['module_name']
    file_name = resp['file_name']
    
    #==============================================================================
    # Get standard sample
    #==============================================================================
    url_to_append = '/api/sample/normalize/{0}'.format(project_id)
    body = {
            'data_params': {
                            'module_name': module_name,
                            'file_name': file_name
                            },
            'module_params':{
                            'sampler_module_name': 'standard'               
                            },
            'sample_params':{
                            'restrict_to_selected': True, # restrict to selected columns
                            'randomize': True
                            }
            }
    resp = post_resp(url_to_append, body)
    
    #=============================================================================
    # Select columns
    #==============================================================================
    url_to_append = '/api/normalize/select_columns/{0}'.format(project_id)
    body = params['select_columns']
    resp = post_resp(url_to_append, body)
    
    #==============================================================================
    # Schedule infer MVS
    #==============================================================================
    url_to_append = '/api/schedule/infer_mvs/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'INIT', 'file_name': file_name}
            }
    resp = post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    infer_mvs_resp = wait_get_resp(url_to_append)

    #==============================================================================
    # Get MVS-specific sample
    #==============================================================================
    
    url_to_append = '/api/sample/normalize/{0}'.format(project_id)
    body = {
            'data_params': {
                            'module_name': module_name,
                            'file_name': file_name
                            },
            'module_params':{
                            'sampler_module_name': 'sample_mvs' ,
                            'module_params': infer_mvs_resp['result']
                            }
            }
    resp = post_resp(url_to_append, body)
    
    #==============================================================================
    # Schedule replace MVS
    #==============================================================================
    url_to_append = '/api/schedule/replace_mvs/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'INIT', 'file_name': file_name},
            'module_params': infer_mvs_resp['result']
            }
    resp = post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = wait_get_resp(url_to_append)
    
    # TODO: add normalize here
    
    #==============================================================================
    # Schedule _concat_with_init
    #==============================================================================
    url_to_append = '/api/schedule/concat_with_init/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'INIT', 'file_name': file_name}
            }
    resp = post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = wait_get_resp(url_to_append, max_wait=10)
    
    
    #==============================================================================
    # Read normalize metadata
    #==============================================================================
    url_to_append = '/api/metadata/normalize/{0}'.format(project_id)
    resp = get_resp(url_to_append)


    return project_id


def link_pipeline(params):
    #==============================================================================
    # Create new link project
    #==============================================================================
    url_to_append = '/api/new/link'
    body = params['new_project']
    resp = post_resp(url_to_append, body)
    project_id = resp['project_id']
    
    
    #==============================================================================
    # Add projects to use as source and ref
    #==============================================================================
    url_to_append = '/api/link/select_file/{0}'.format(project_id)
    body = {'file_role': 'source',
            'project_id': params['source_project_id']}
    resp = post_resp(url_to_append, body)
    
    body = {'file_role': 'ref',
            'project_id': params['ref_project_id']}
    resp = post_resp(url_to_append, body)
    
    
    #==============================================================================
    # Add column matches
    #==============================================================================
    url_to_append = '/api/link/add_column_matches/{0}/'.format(project_id)
    body = params['column_matches']
    resp = post_resp(url_to_append, body)
    
    #==============================================================================
    # Add certain column matches
    #==============================================================================
    url_to_append = '/api/link/add_column_certain_matches/{0}/'.format(project_id)
    body = params['column_certain_matches']
    resp = post_resp(url_to_append, body)
    
    # TODO: Add method to automatically add training data
    
    #==============================================================================
    # Add training_data  
    #
    # WARNING: In interface, training data is generated during labelling
    #
    #==============================================================================
    url_to_append = '/api/upload_config/link/{0}/'.format(project_id)
    training_file_path = params['training_file_path']
    with open(training_file_path) as f:
        training_params = json.load(f)
    body = {'data_params': {
                            "module_name": 'dedupe_linker',
                            "file_name": 'training.json'
                            },
            'module_params': training_params}
    resp = post_resp(url_to_append, body)
    
    #==============================================================================
    # Create labeller
    #==============================================================================
    url_to_append = '/api/schedule/create_labeller/{0}/'.format(project_id)
    body = {}
    resp = post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = wait_get_resp(url_to_append, max_wait=20)
    
    #==============================================================================
    # Run linker
    #==============================================================================
    url_to_append = '/api/schedule/linker/{0}/'.format(project_id)
    body = {}
    resp = post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = wait_get_resp(url_to_append, max_wait=600)

    return project_id





if __name__ == '__main__':
    
    #==============================================================================
    # RUN NORMALIZE PIPELINE ON SOURCE
    #==============================================================================
    params = {
        'new_project': {'description': 'File to use as source for testing',
                        'display_name': 'test_source.csv',
                        'internal': False},
        'file_path': 'local_test_data/integration_1/source.csv',
        'select_columns':{
                        'columns': ['id_cordee', 'uai', 'commune', 'academie', 
                                    'lycees_sources', 'departement', 'region']
                        }
    }
    source_project_id = normalize_pipeline(params)
    
    #==============================================================================
    # RUN NORMALIZE PIPELINE ON REF
    #==============================================================================
    params = {
        'new_project': {'description': 'File to use as ref for testing',
                    'display_name': 'test_ref.csv',
                    'internal': True},
        'file_path': 'local_test_data/integration_1/ref.csv',
        'select_columns':{
                        'columns': ['full_name', 'adresse_uai', 'localite_acheminement_uai', 
                            'patronyme_uai', 'departement', 'code_postal_uai', 'numero_uai']
                        }
    }
    ref_project_id = normalize_pipeline(params)
    
    #==============================================================================
    # RUN LINK PIPELINE
    #==============================================================================
    params = {
                'source_project_id': source_project_id,
                'ref_project_id': ref_project_id,
                'new_project': {'description': 'Link project for testing',
                                'display_name': 'test_link',
                                'internal': False},
                'column_matches': {
                            'column_matches':
                                [{"ref": ["departement"], "source": ["departement"]}, 
                                {"ref": ["full_name"], "source": ["lycees_sources"]}, 
                                {"ref": ["localite_acheminement_uai"], "source": ["commune"]}]
                        },
                'column_certain_matches': {
                             'column_certain_matches': {'source': 'uai', 'ref': 'numero_uai'}
                        },
                'training_file_path': 'local_test_data/integration_1/training.json'
    }
    link_project_id = link_pipeline(params)
    
    #==============================================================================
    # Delete projects   
    #==============================================================================
    for _id in [source_project_id, ref_project_id]:
        url_to_append = '/api/delete/normalize/{0}'.format(_id)
        resp = get_resp(url_to_append)
    
    url_to_append = '/api/delete/link/{0}'.format(link_project_id)
    resp = get_resp(url_to_append)

    #==============================================================================
    # List projects
    #==============================================================================
    url_to_append = '/api/projects/link'
    resp = get_resp(url_to_append)


   
    
    