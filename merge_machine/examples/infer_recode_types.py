#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:00:14 2017

@author: m75380

API use example to infer column types and recode values accordingly

"""

import json
import requests
import time

PROTOCOL = 'http://'
HOST = '0.0.0.0:5000'

# CSV file to recode
file_path = '../local_test_data/source.csv'

# Paths of files to write
inference_file_path_write = 'infer_types.json'
output_file_path = 'source_recoded.csv'


# =============================================================================
# Create new project
# =============================================================================

url = PROTOCOL + HOST + '/api/new/normalize'
body = {'description': 'Test project for type inference and recoding on a file'}
resp = requests.post(url, json=body)
parsed_resp = json.loads(resp.content.decode())

# Retrieve the project_id of the newly created project
project_id = parsed_resp['project_id']

# =============================================================================
# Upload file
# =============================================================================

url = PROTOCOL + HOST + '/api/normalize/upload/{0}'.format(project_id)
with open(file_path, 'rb') as f:
    resp = requests.post(url, files={'file': f})
parsed_resp = json.loads(resp.content.decode())

# =============================================================================
# Schedule type_inference on columns
# =============================================================================

# Module from which to fetch the file (INIT)
module_name = parsed_resp['run_info']['module_name']
# file_name of the file to fetch from which to fetch the file (here: INIT)
file_name = parsed_resp['run_info']['file_name']
# data_params: information on the file to use as input for inference or transformation
data_params = {'module_name': module_name, 'file_name': file_name}
# module_params: parameters to use for inference or transform (here: None)
module_params = None

url = PROTOCOL + HOST + '/api/schedule/infer_types/{0}/'.format(project_id)
body =  {'data_params': data_params, 'module_params': module_params}
resp = requests.post(url, json=body)
parsed_resp = json.loads(resp.content.decode())

# =============================================================================
# Wait for response for type_inference job
# =============================================================================

# In the response of a scheduled job, the 'job_result_api_url' field is the 
# url to request to get the status and rsult of the job
url = PROTOCOL + HOST + parsed_resp['job_result_api_url']
start_time = time.time()
# Wait for a maximimum of 100 seconds
while (time.time() - start_time) <= 100:
    resp = requests.get(url)
    parsed_resp = json.loads(resp.content.decode())

    # Check if job was completed
    if parsed_resp['completed']:
        break
    
    # Otherwise check again in 0.5 second
    time.sleep(0.5)
    
# =============================================================================
# Write output to file
# =============================================================================
# NB: Currently, infer_types returns many superflueous information. The interesting
# information is in parsed_resp['result']['column_types']
with open(inference_file_path_write, 'w') as w:
    json.dump(parsed_resp['result']['column_types'], w)
print('Result of type inference written to:', inference_file_path_write)

# =============================================================================
# Schedule recode_type (module is still being worked on)
# =============================================================================
# data_params: information on the file to use as input for inference or transformation
# NB: file_name and module_name to use as input did not change²
data_params = {'module_name': module_name, 'file_name': file_name}
# module_params: parameters to use for inference or transform (here: we use the
# result of the previous inference step)

module_params = parsed_resp['result']

url = PROTOCOL + HOST + '/api/schedule/recode_types/{0}/'.format(project_id)
body =  {'data_params': data_params, 'module_params': module_params}
resp = requests.post(url, json=body)
parsed_resp = json.loads(resp.content.decode())

# =============================================================================
# Wait for response for recode_types job
# =============================================================================
url = PROTOCOL + HOST + parsed_resp['job_result_api_url']
start_time = time.time()
# Wait for a maximimum of 100 seconds
while (time.time() - start_time) <= 100:
    resp = requests.get(url)
    parsed_resp = json.loads(resp.content.decode())

    # Check if job was completed
    if parsed_resp['completed']:
        break
    
    # Otherwise check again in 0.5 second
    time.sleep(0.5)
    
# =============================================================================
# Download the transformed file
# =============================================================================
url = PROTOCOL + HOST + '/api/download/normalize/{0}'.format(project_id)

# Get the file_name and module_name to which the transformed file was written
module_name = parsed_resp['result']['module_name']
file_name = parsed_resp['result']['file_name']
data_params = {'module_name': module_name, 'file_name': file_name}
module_params = None

body = {
        'data_params': data_params,
        'module_params': module_params
        }
resp = requests.post(url, json=body)

with open(output_file_path,'wb') as w:
    w.write(resp.content)
    
print('Result file of recode_types written to:', inference_file_path_write)