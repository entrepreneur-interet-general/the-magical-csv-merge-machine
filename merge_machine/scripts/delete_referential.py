#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:43:08 2017

@author: m75380

Script to add SIRENE to API
"""
import argparse
import json
import os

from api_helpers import APIConnection


# =============================================================================
# Paths to configuration files
# =============================================================================

# Default
display_name = 'test_ref.csv'
project_id = None

# Path to configuration
connection_config_path = os.path.join('conf', 'local_connection_parameters.json')
logs_path = 'logs.json'

# =============================================================================
# Get arguments from argparse
# =============================================================================
parser = argparse.ArgumentParser(description='Delete a reference from  its' \
                                + ' project_id or display name')
parser.add_argument('--name', 
                    help='Display name of the file to delete',
                    default=display_name)
parser.add_argument('--proj-id', 
                    help='Project id of the file to delete (use instead of ' \
                        + 'display_name). ',
                    default=project_id)
parser.add_argument('--conn', 
                    help='Path to the json configuration file that' \
                    + ' with information on the connection to the API',
                    default=connection_config_path)
parser.add_argument('--logs', 
                    help='Path to the json log file',
                    default=logs_path)
args = parser.parse_args()

display_name = args.name
project_id = args.proj_id
connection_config_path = args.conn
logs_path = args.logs

# =============================================================================
# Check that we are selecting either by project_id or by display name
# =============================================================================
if project_id is not None:
    display_name = None
assert int(display_name is None) + int(project_id is None) == 1

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
# Delete projects
# =============================================================================
if display_name is None:
    url_to_append = '/api/delete/normalize/{0}'.format(project_id)
    resp = c.get_resp(url_to_append)
else:
    for metadata in filter(lambda x: x['display_name']==display_name, resp):
        old_project_id = metadata['project_id']
        url_to_append = '/api/delete/normalize/{0}'.format(old_project_id)
        resp = c.get_resp(url_to_append)
        
# =============================================================================
# Remove old project from logs if present
# =============================================================================
if display_name is None:
    display_name = [key for key, value in logs.items() if value==project_id]
    if display_name:
        assert len(display_name) == 1
        display_name = display_name[0]

if display_name:
    if display_name in logs:
        del logs[display_name]
        with open(logs_path, 'w') as w:
            json.dump(logs, w)