#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:18 2017

@author: leo
"""

import json
import requests
import time

PROTOCOL = 'http://'
HOST = '0.0.0.0:5000'
PRINT = True

def _print(url_to_append, parsed_resp):
    if PRINT:
        print('\n>>>>>>>>>>>>>>>>>>>>>>>>>\n', url_to_append, '\n\n', parsed_resp)
            
def get_resp(url_to_append):
    url = PROTOCOL + HOST + url_to_append
    resp = requests.get(url)
    
    if resp.ok:
        parsed_resp = json.loads(resp.content.decode())
        _print(url_to_append, parsed_resp)
        return parsed_resp
    else: 
        raise Exception('Problem:\n', resp)
        
def post_resp(url_to_append, body, **kwargs):
    url = PROTOCOL + HOST + url_to_append
    resp = requests.post(url, json=body, **kwargs)     
    
    if resp.ok:
        parsed_resp = json.loads(resp.content.decode())
        _print(url_to_append, parsed_resp)
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
            print('\n--> Waited {0} seconds'.format(time.time()-start_time))
            print('\n', parsed_resp)
            return parsed_resp
        time.sleep(0.25)
    print(time.time() - start_time)
    raise Exception('Timed out for {0}'.format(url_to_append))


#==============================================================================
# List projects
#==============================================================================
url_to_append = '/api/projects/normalize'
resp = get_resp(url_to_append)


#def normalize_pipe():
#==============================================================================
# Create new_normalization project
#==============================================================================
url_to_append = '/api/new/normalize'
body = {'description': 'File to use as source for testing',
        'display_name': 'test_source.csv',
        'internal': False}
resp = post_resp(url_to_append, body)
project_id = resp['project_id']

#==============================================================================
# Upload new file
#==============================================================================
url_to_append = '/api/normalize/upload/{0}'.format(project_id)
file_path = 'local_test_data/source.csv'
# file_name = file_path.rsplit('/', 1)[-1]
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
file_name = resp['file_name']

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
resp = wait_get_resp(url_to_append)

#==============================================================================
# Schedule replace MVS
#==============================================================================
url_to_append = '/api/schedule/replace_mvs/{0}/'.format(project_id)
body = {
        'data_params': {'module_name': 'INIT', 'file_name': file_name},
        'module_params': resp['result']
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


#==============================================================================
# Delete project    
#==============================================================================
url_to_append = '/api/delete/normalize/{0}'.format(project_id)
resp = get_resp(url_to_append)
   
    
    
    
    
    
    
    
    
assert False
    
# Select id
_id = '4e8286f034eef40e89dd99ebe6d87f21'

# Get metadata
to_append = 'project/metadata/{0}'.format(_id)
url = protocol + host + to_append
resp = requests.post(url)

if resp.ok:
    parsed_resp = json.loads(resp.content.decode())
else: 
    raise Exception('Problem with metadata')

# Upload a reference
#to_append = 'project/upload/{0}'.format(_id)
#url = protocol + host + to_append


# Infer Missing values
to_append = 'project/modules/infer_mvs/{0}'.format(_id)
url = protocol + host + to_append
resp = requests.post(url)

if resp.ok:
    parsed_resp = json.loads(resp.content.decode())
else: 
    raise Exception('Problem with infer_mvs')
    
# Replace Missing Values
to_append = 'project/modules/replace_mvs/{0}'.format(_id)
url = protocol + host + to_append
body = {'params': parsed_resp['response']}
resp = requests.post(url, json=body)   

if resp.ok:
    parsed_resp = json.loads(resp.content.decode())
else: 
    raise Exception('Problem with replace_mvs')
    

# Apply dedupe_linker
to_append = 'project/link/dedupe_linker/{0}'.format(_id)
url = protocol + host + to_append
body = json.load(open('sample_dedupe_link_request.json'))
resp = requests.post(url, json=body)  
 
if resp.ok:
    parsed_resp = json.loads(resp.content.decode())
else: 
    raise Exception('Problem with dedupe_linker')