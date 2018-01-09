#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:08 2017

@author: m75380

Script to add SIRENE to API
"""
from datetime import datetime
import json
import os

from api_helpers import APIConnection

# Path to configuration
config_path = os.path.join('parameters', 'sirene.json')
connection_config_path = os.path.join('parameters', 'local_connection_parameters.json')

# =============================================================================
# Define how to connect to API
# =============================================================================
conn_params = json.load(open(connection_config_path))
PROTOCOL = conn_params['PROTOCOL']
HOST = conn_params['HOST']
PRINT = conn_params['PRINT']
PRINT_CHAR_LIMIT = conn_params['PRINT_CHAR_LIMIT']

c = APIConnection(PROTOCOL, HOST, PRINT, PRINT_CHAR_LIMIT)

# =============================================================================
# Load parameters for pipeline
# =============================================================================
params = json.load(open(config_path))

# Add date to description 
if 'description' not in params['new_project']:
    params['new_project']['description'] = ''
params['new_project']['description'] += ' ({0})'.format(datetime.now().isoformat()[:10])

#==============================================================================
# Create new normalization project
#==============================================================================
url_to_append = '/api/new/normalize'
body = params['new_project']
resp = c.post_resp(url_to_append, body)
project_id = resp['project_id']

#==============================================================================
# Upload new file
#==============================================================================
url_to_append = '/api/normalize/upload/{0}'.format(project_id)
file_path = params['file_path']
with open(file_path, 'rb') as f:
    resp = c.post_resp(url_to_append, 
                     body, 
                     files={'file': f})

#==============================================================================
# Get last written
#==============================================================================
module_name = 'INIT'
file_name = os.path.split(params['file_path'])[-1]

# =============================================================================
# Skip replace_mvs, recode_types, concat_with_init
# =============================================================================
url_to_append = '/api/set_skip/normalize/{0}'.format(project_id)
for mn in ['replace_mvs', 'recode_types', 'concat_with_init']:
    body = {'data_params': {'module_name': mn, 'file_name': file_name},
            'module_params': {'skip_value': True}}
    resp = c.post_resp(url_to_append, body)

# =============================================================================
# Index reference     
# =============================================================================
url_to_append = '/api/schedule/create_es_index/{0}/'.format(project_id)
body = {
        'data_params': {
                        'project_type': 'normalize', 
                        'module_name':module_name, 
                        'file_name': file_name
                        },
        'module_params': {'columns_to_index': params.get('columns_to_index', None),
                          'force': True}
        }
resp = c.post_resp(url_to_append, body)
job_id = resp['job_id']

#==============================================================================
# --> Wait for job result
#==============================================================================
url_to_append = '/queue/result/{0}'.format(job_id)
resp = c.wait_get_resp(url_to_append, max_wait=10000)