#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:08 2017

@author: m75380

Script to add SIRENE to API
"""

from api_helpers import APIConnection

# =============================================================================
# Define how to connect to API
# =============================================================================
PROTOCOL = 'http://'
HOST = '127.0.0.1:5000'
PRINT = True
PRINT_CHAR_LIMIT = 10000

c = APIConnection(PROTOCOL, HOST, PRINT, PRINT_CHAR_LIMIT)


# =============================================================================
# Load parameters for pipeline
# =============================================================================

params = {
            'file_path': '/home/m75380/Documents/eig/the-magical-csv-merge-machine/merge_machine/local_test_data/sirene/sirene_filtered.csv'
        }

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
url_to_append = '/api/last_written/normalize/{0}'.format(project_id)
body = {
        'before_module': 'replace_mvs'        
        }
resp = c.post_resp(url_to_append, body)
module_name = resp['module_name']
file_name = resp['file_name']

# =============================================================================
# Skip replace_mvs
# =============================================================================
url_to_append = '/api/set_skip/<project_type>/<project_id>'
for module_name in ['replace_mvs', 'recode_types', 'concat_with_init']:
    body = {'data_params': {'module_name': module_name, 'file_name': file_name}}
    resp = c.post_resp(url_to_append)

# =============================================================================
# Index reference     
# =============================================================================
url_to_append = '/api/schedule/create_es_index/{0}/'.format(project_id)
body = {
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