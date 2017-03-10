#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:18 2017

@author: leo
"""

import json
import requests

protocol = 'http://'
host = '127.0.0.1:5000/'


# List projects
to_append = 'admin/list_projects'
url = protocol + host + to_append
resp = requests.get(url)

if resp.ok:
    parsed_resp = json.loads(resp.content.decode())
else: 
    print('Problem with list projects')
    
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