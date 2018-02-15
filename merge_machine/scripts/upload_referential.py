#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:08 2017

@author: m75380

Script to add SIRENE to API
"""
import argparse
from datetime import datetime
import json
import os

from api_helpers import APIConnection

# =============================================================================
# Paths to configuration files
# =============================================================================

# Default
config_path = os.path.join('conf', 'rnsr.json')
connection_config_path = os.path.join('conf', 'local_connection_parameters.json')
logs_path = 'logs.json'

# =============================================================================
# Get arguments from argparse
# =============================================================================
parser = argparse.ArgumentParser(description='Upload and index' \
                                 + ' referential to the API service.')
parser.add_argument('--conf', 
                    help='Path to the json configuration file that' \
                    + ' with information on the file to upload',
                    default=config_path)
parser.add_argument('--conn', 
                    help='Path to the json configuration file that' \
                    + ' with information on the connection to the API',
                    default=connection_config_path)
parser.add_argument('--logs', 
                    help='Path to the json log file',
                    default=logs_path)
args = parser.parse_args()

config_path = args.conf
connection_config_path = args.conn
logs_path = args.logs

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
# Get additional info if available
if os.path.isfile(params.get('info_file_path', '')):
    params['new_project']['info'] = json.load(open(params['info_file_path']))

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

# =============================================================================
# Load logs
# =============================================================================
if os.path.isfile(logs_path):
    logs = json.load(open(logs_path))
else:
    logs = dict()

#==============================================================================
# Fetch public projects
#==============================================================================
url_to_append = '/api/public_projects/normalize'
resp = c.get_resp(url_to_append)

# =============================================================================
# Delete project with same display_name if necessary
# =============================================================================
for metadata in filter(lambda x: x['display_name']==params['new_project']['display_name'], resp):
    old_project_id = metadata['project_id']
    if old_project_id != project_id:
        url_to_append = '/api/delete/normalize/{0}'.format(old_project_id)
        resp = c.get_resp(url_to_append)
    
# =============================================================================
# Add new project to logs
# =============================================================================
logs[params['new_project']['display_name']] = project_id
with open(logs_path, 'w') as w:
    json.dump(logs, w)
