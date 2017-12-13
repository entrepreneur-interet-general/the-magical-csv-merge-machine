#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:18 2017

@author: leo
"""

import json
import os

from api_helpers import APIConnection


PROTOCOL = 'http://'
HOST = '127.0.0.1:5000'
PRINT = True
PRINT_CHAR_LIMIT = 10000

c = APIConnection(PROTOCOL, HOST, PRINT, PRINT_CHAR_LIMIT)



def normalize_pipeline(c, params):
    '''
    INPUT:
        c: instance of APICO=onnection
        params: ...
    '''
    
    #==============================================================================
    # Create new normalization project
    #==============================================================================
    url_to_append = '/api/new/normalize'
    #    body = {'description': 'File to use as source for testing',
    #            'display_name': 'test_source.csv',
    #            'public': False}
    body = params['new_project']
    resp = c.post_resp(url_to_append, body)
    project_id = resp['project_id']
    
    #==============================================================================
    # Upload new file
    #==============================================================================
    url_to_append = '/api/normalize/upload/{0}'.format(project_id)
    # file_path = 'local_test_data/integration_1/source.csv'
    # file_name = file_path.rsplit('/', 1)[-1]
    file_path = params['file_path']
    with open(file_path, 'rb') as f:
        resp = c.post_resp(url_to_append, 
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
    resp = c.post_resp(url_to_append, body)
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
                            },
            '__test': '__has_underscores_before_and_after__'
            }
    resp = c.post_resp(url_to_append, body)
    
    #=============================================================================
    # Select columns
    #==============================================================================
    url_to_append = '/api/normalize/select_columns/{0}'.format(project_id)
    body = params['select_columns']
    resp = c.post_resp(url_to_append, body)
    
    #==============================================================================
    # Schedule infer MVS
    #==============================================================================
    url_to_append = '/api/schedule/infer_mvs/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'INIT', 'file_name': file_name}
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    infer_mvs_resp = c.wait_get_resp(url_to_append, max_wait=100)

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
    resp = c.post_resp(url_to_append, body)
    
    #==============================================================================
    # Schedule replace MVS
    #==============================================================================
    url_to_append = '/api/schedule/replace_mvs/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'INIT', 'file_name': file_name},
            'module_params': infer_mvs_resp['result']
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append)
    
    #==============================================================================
    # Schedule infer Types
    #==============================================================================
    url_to_append = '/api/schedule/infer_types/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'replace_mvs', 'file_name': file_name}
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    infer_types_resp = c.wait_get_resp(url_to_append, 100)

    #==============================================================================
    # Schedule recode types
    #==============================================================================
    url_to_append = '/api/schedule/recode_types/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'replace_mvs', 'file_name': file_name},
            'module_params': infer_types_resp['result']
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    infer_mvs_resp = c.wait_get_resp(url_to_append)
        
    #==============================================================================
    # Schedule _concat_with_init
    #==============================================================================
    url_to_append = '/api/schedule/concat_with_init/{0}/'.format(project_id)
    body = {
            'data_params': {'module_name': 'replace_mvs', 'file_name': file_name}
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # Run all transforms on original file
    #==============================================================================
    file_name = file_name.replace('MINI__', '')
    url_to_append = '/api/schedule/run_all_transforms/{0}/'.format(project_id)
    body = {
            'data_params': {'file_name': file_name}
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']        
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append, max_wait=180)
        
    
    #==============================================================================
    # Read normalize metadata
    #==============================================================================
    url_to_append = '/api/metadata/normalize/{0}'.format(project_id)
    resp = c.get_resp(url_to_append)


    #==============================================================================
    # Download file    
    #==============================================================================
    url_to_append = '/api/download/normalize/{0}'.format(project_id)
    body = {
            'data_params': {
                'module_name': 'concat_with_init'},
            'module_params': {
                'file_type': 'csv'}
            }
    c.print_ = False
    resp = c.post_download(url_to_append, body)    
    c.print_ = True

    return project_id


def link_pipeline(c, params):
    '''
    INPUT:
        c: instance of APICO=onnection
        params: ...
    '''
    
    
    #==============================================================================
    # Create new link project
    #==============================================================================
    url_to_append = '/api/new/link'
    body = params['new_project']
    resp = c.post_resp(url_to_append, body)
    project_id = resp['project_id']
        
    
    #==============================================================================
    # Add projects to use as source and ref
    #==============================================================================
    url_to_append = '/api/link/select_file/{0}'.format(project_id)
    body = {'file_role': 'source',
            'project_id': params['source_project_id']}
    resp = c.post_resp(url_to_append, body)
    
    body = {'file_role': 'ref',
            'project_id': params['ref_project_id']}
    resp = c.post_resp(url_to_append, body)

    # =============================================================================
    # Index reference     
    # =============================================================================
    url_to_append = '/api/schedule/create_es_index/{0}/'.format(project_id)
    body = {
            'module_params': {'columns_to_index': params.get('columns_to_index_ref', None),
                              'force': True}
            }
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append, max_wait=10000)

    
    #==============================================================================
    # Add column matches
    #==============================================================================
    url_to_append = '/api/link/add_column_matches/{0}/'.format(project_id)
    body = params['column_matches']
    resp = c.post_resp(url_to_append, body)
    
    #==============================================================================
    # Add certain column matches
    #==============================================================================
    url_to_append = '/api/link/add_column_certain_matches/{0}/'.format(project_id)
    body = params.get('column_certain_matches')
    if body is not None:
        resp = c.post_resp(url_to_append, body)
    
    # TODO: Add method to automatically add training data
    
    #==============================================================================
    # Add training_data  
    #
    # WARNING: In interface, training data is generated during labelling
    #
    #==============================================================================
    
    # TODO: what is this ????
    url_to_append = '/api/upload_config/link/{0}/'.format(project_id)
    es_learned_settings_file_path = params['es_learned_settings_file_path']
    with open(es_learned_settings_file_path) as f:
        es_learned_settings = json.load(f)
    body = {'data_params': {
                            "module_name": 'es_linker',
                            "file_name": 'learned_settings.json'
                            },
            'module_params': es_learned_settings}
    resp = c.post_resp(url_to_append, body)
    
    
    #    #==============================================================================
    #    # Infer restriction parameters
    #    #==============================================================================
    #    url_to_append = '/api/schedule/infer_restriction/{0}/'.format(project_id)
    #    body = {}
    #    resp = c.post_resp(url_to_append, body)
    #    job_id = resp['job_id']
    #    
    #    #==============================================================================
    #    # --> Wait for job result
    #    #==============================================================================
    #    url_to_append = '/queue/result/{0}'.format(job_id)
    #    infer_restriction_resp = c.wait_get_resp(url_to_append, max_wait=20)
    
    #    #==============================================================================
    #    # Perform restriction
    #    #==============================================================================
    #    url_to_append = '/api/schedule/perform_restriction/{0}/'.format(project_id)
    #    body = {
    #            'module_params': infer_restriction_resp['result']
    #            }
    #    resp = c.post_resp(url_to_append, body)
    #    job_id = resp['job_id']
    #
    #    #==============================================================================
    #    # --> Wait for job result
    #    #==============================================================================
    #    url_to_append = '/queue/result/{0}'.format(job_id)
    #    resp = c.wait_get_resp(url_to_append)    
    
    #==============================================================================
    # Create labeller
    #==============================================================================
    url_to_append = '/api/schedule/create_es_labeller/{0}/'.format(project_id)
    body = {}
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']
    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append, max_wait=20)
    
    #==============================================================================
    # Run linker
    #==============================================================================
    url_to_append = '/api/schedule/es_linker/{0}/'.format(project_id)
    body = {'module_params': es_learned_settings}
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']

    #    resp = c.post_resp(url_to_append, body)
    #    job_id_useless = resp['job_id']
    
    # Cancel job   
    #    url_to_append = '/queue/cancel/{0}'.format(job_id_useless)
    #    resp = c.get_resp(url_to_append)
    
    # Check that job was cancelled

    
    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append, max_wait=600)
    
    # =============================================================================
    # Index result of linking     
    # =============================================================================
    #    url_to_append = '/api/schedule/create_es_index/{0}/'.format(project_id)
    #    body = {
    #            'module_params': {'columns_to_index_is_none': None,
    #                              'for_linking': False,
    #                              'force': True}
    #            }
    #    resp = c.post_resp(url_to_append, body)
    #    job_id = resp['job_id']
    #
    #    #==============================================================================
    #    # --> Wait for job result
    #    #==============================================================================
    #    url_to_append = '/queue/result/{0}'.format(job_id)
    #    resp = c.wait_get_resp(url_to_append, max_wait=10000)
    
    
    #==============================================================================
    # --> Analyze results
    #==============================================================================
    url_to_append = '/api/schedule/link_results_analyzer/{0}/'.format(project_id)
    body = {'data_params': {
                            "module_name": 'es_linker',
                            "file_name": params['source_file_name'].rsplit('.')[0] + '.csv'
                            }
            }    
    resp = c.post_resp(url_to_append, body)
    job_id = resp['job_id']    

    #==============================================================================
    # --> Wait for job result
    #==============================================================================
    url_to_append = '/queue/result/{0}'.format(job_id)
    resp = c.wait_get_resp(url_to_append, max_wait=20)    

    return project_id


if __name__ == '__main__':
    
    # Change current path to path of test_file.py		
    curdir = os.path.dirname(os.path.realpath(__file__))		
    os.chdir(curdir)
    
    
    # Define how to connect to API
    PROTOCOL = 'http://'
    HOST = '127.0.0.1:5000'
    PRINT = True
    PRINT_CHAR_LIMIT = 10000

    c = APIConnection(PROTOCOL, HOST, PRINT, PRINT_CHAR_LIMIT)
    
    # Parse user request for testing
    import argparse 
    
    parser = argparse.ArgumentParser(description='Run normalization and link pipelines')
    parser.add_argument('--dir', 
                        type=str, 
                        default=os.path.join('local_test_data', 'integration_1'), 
                        help='Path to directory containing test data')
    parser.add_argument('--keep', 
                        action='store_true',
                        help='Use this flag to NOT delete projects after testing')
    parser.add_argument('--source', default=None, help='ID of source to skip pipeline')
    parser.add_argument('--ref', default=None, help='ID of ref to skip pipeline')
    args = parser.parse_args()
    
    # Parameters
    source_params_path = os.path.join(args.dir, 'source_params.json')
    ref_params_path = os.path.join(args.dir, 'ref_params.json')
    link_params_path = os.path.join(args.dir, 'link_params.json')
    
    #==============================================================================
    # RUN NORMALIZE PIPELINE ON SOURCE
    #==============================================================================        
    with open(source_params_path) as f:
        source_params = json.load(f)
    source_params['file_path'] = os.path.join(args.dir, source_params['file_name'])
    if args.source is None:
        source_project_id = normalize_pipeline(c, source_params)
    else:
        source_project_id = args.source
    
    #==============================================================================
    # RUN NORMALIZE PIPELINE ON REF
    #==============================================================================
    with open(ref_params_path) as f:
        ref_params = json.load(f)
    ref_params['file_path'] = os.path.join(args.dir, ref_params['file_name'])
    
    if args.ref is None:
        ref_project_id = normalize_pipeline(c, ref_params)
    else:
        ref_project_id = args.ref
        
    #==============================================================================
    # RUN LINK PIPELINE
    #==============================================================================
    with open(link_params_path) as f:
        link_params = json.load(f)                     

    link_params['source_project_id'] = source_project_id
    link_params['ref_project_id'] = ref_project_id

    link_params['source_file_name'] = source_params['file_name']
    
    link_params['es_learned_settings_file_path'] = os.path.join(args.dir, link_params['es_learned_settings_file_path'])
    link_params['training_file_path'] = os.path.join(args.dir, link_params['training_file_name'])
               
    link_project_id = link_pipeline(c, link_params)
    
    #==============================================================================
    # Delete projects   
    #==============================================================================
    if not args.keep:
        for (type_, id_) in [('normalize', source_project_id), 
                             ('normalize', ref_project_id), 
                             ('link', link_project_id)]:
            url_to_append = '/api/delete/{0}/{1}'.format(type_, id_)
            resp = c.get_resp(url_to_append)

    #==============================================================================
    # List projects
    #==============================================================================
    url_to_append = '/api/public_project_ids/link'
    resp = c.get_resp(url_to_append)

    print('source_project_id:', source_project_id)
    print('ref_project_id:', ref_project_id)
    print('link_project_id:', link_project_id)   
    
    sh_com = 'testapi --dir {0} --source {1} --ref {2} --keep'.format(args.dir, source_project_id, ref_project_id)
    print('Re-run link with:\n', sh_com)
    