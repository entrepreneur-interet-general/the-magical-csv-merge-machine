#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:08 2017

@author: m75380

Script to add SIRENE to API
"""
from datetime import datetime
import os

from api_helpers import APIConnection

# =============================================================================
# Define how to connect to API
# =============================================================================
PROTOCOL = 'http://'
HOST = '51.15.221.77:5000'
PRINT = True
PRINT_CHAR_LIMIT = 10000

c = APIConnection(PROTOCOL, HOST, PRINT, PRINT_CHAR_LIMIT)


# =============================================================================
# Load parameters for pipeline
# =============================================================================

params = {
          "new_project": {
                        "description": "Base SIRENE ({0})".format(datetime.now().isoformat()[:10]),
                        "display_name": "SIRENE",
                        "public": True
                      },
        
            'file_path': '/home/m75380/Documents/eig/the-magical-csv-merge-machine/merge_machine/local_test_data/sirene/sirene_filtered.csv',
            
            'columns_to_index': {
                                     'CEDEX': [],
                                     'ENSEIGNE': ['french', 'n_grams', 'integers', 'city'],
                                     'L1_DECLAREE': ['french', 'n_grams', 'integers', 'city'],
                                     'L1_NORMALISEE': ['french', 'n_grams', 'integers', 'city'],
                                     'L4_DECLAREE': ['french', 'n_grams', 'integers', 'city'],
                                     'L4_NORMALISEE': ['french', 'n_grams', 'integers', 'city'],
                                     'L6_DECLAREE': ['french', 'n_grams', 'integers', 'city'],
                                     'L6_NORMALISEE': ['french', 'n_grams', 'integers', 'city'],
                                     'LIBAPET': [],
                                     'LIBCOM': ['french', 'n_grams', 'city'],
                                     'NIC': [],
                                     'NOMEN_LONG': ['french', 'n_grams', 'integers', 'city'],
                                     'PRODEN': [],
                                     'PRODET': [],
                                     'SIEGE': [],
                                     'SIREN': [],
                                     'SIRET': []
                                }
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