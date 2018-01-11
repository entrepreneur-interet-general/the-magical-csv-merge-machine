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

# Project id or display name to delete
display_name = 'RNSR'
project_id = None

# Path to configuration
connection_config_path = os.path.join('conf', 'local_connection_parameters.json')
logs_path = 'logs.json'

# =============================================================================
# Check that we are selecting either by project_id or by display name
# =============================================================================
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

# =============================================================================
# Delete project with same display_name or project id
# =============================================================================
if display_name is not None:
    project_id = logs[display_name]
    
url_to_append = '/api/delete/normalize/{0}'.format(project_id)
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
    del logs[display_name]
    with open(logs_path, 'w') as w:
        json.dump(logs, w)