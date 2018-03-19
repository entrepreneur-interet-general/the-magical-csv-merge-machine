#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo
"""
import argparse
import json
import os
#import sys

# Change directory to main folder to be able to access methods
#main_dir_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
#sys.path.append(main_dir_path)
#os.chdir(main_dir_path)
#print(main_dir_path)
#
#try:
#    from admin import Admin
#except ImportError as e:
#    raise ImportError(e.__str__() + '\ntry running from same directory as file')    
    
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
parser = argparse.ArgumentParser(description='List or delete projects based on '
                         'time of last use or creation. This script will'
                         ' also delete loose normalization project and indices.',
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('request', 
                    type=str,
                    choices=['list', 'delete', 'clean'],
                    help='Choose what to do with the results')

parser.add_argument('project_type', 
                    type=str,
                    choices=['link', 'normalize'],
                    help='Type of project to list or delete (put any value if' \
                    ' using "clean")')

parser.add_argument('-pa', '--project_access', 
                    type=str,
                    choices=['all', 'public', 'private'],
                    default='private',
                    nargs='?',
                    help='Restrictions on project access')

parser.add_argument('-a', '--action', 
                    type=str, 
                    choices=['created', 'last_used'],
                    default='last_used',
                    nargs='?',
                    help='Whether to filter on date of creation or last use')

parser.add_argument('-w', '--when',
                   type=str,
                   choices=['before', 'after'],
                   default='before',
                   nargs='?',
                   help='Choose whether to delete before or after the target' 
                        'timestamp')

parser.add_argument('-hfn', '--hours_from_now', 
                    type=float,
                    default=24*14, # 14 days
                    nargs='?',
                    help='Number of last hours for which to keep data')

parser.add_argument('-ki', '--keep_indices',
                    action='store_true',
                    help='Flag to keep Elasticsearch indices despite having'
                    ' deleted the MMM projects')    
parser.add_argument('-kll', '--keep_loose_links',
                    action='store_true',
                    help='Flag to NOT delete loose link projects')    

args = parser.parse_args()

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
# List or delete projects
# =============================================================================

body = {'module_params': 
            {
            'project_access': args.project_access, 
            'action': args.action, 
            'when': args.when, 
            'hours_from_now': args.hours_from_now
            }
        }

# Actual requests
if args.request == 'list':
    url_to_append = '/api/admin/list_projects/{0}'.format(args.project_type)
    resp = c.post_resp(url_to_append, body)
elif args.request == 'delete':
    url_to_append = '/api/admin/remove_projects/{0}'.format(args.project_type)
    resp = c.post_resp(url_to_append, body)
else:
    if args.request != 'clean':
        raise ValueError('No action associated to request: {0}'.format(args.request))

#==============================================================================
# Clean up (delete unused indices or loose links)
#==============================================================================

if (args.request in ['delete', 'clean']):
    if not args.keep_indices:
        url_to_append = '/api/admin/delete_unused_indices'.format(args.project_type)
        resp = c.get_resp(url_to_append)
        
    if not args.keep_loose_links:
        url_to_append = '/api/admin/delete_loose_links'.format(args.project_type)
        resp = c.get_resp(url_to_append)
