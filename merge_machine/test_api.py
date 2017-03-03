#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:48:18 2017

@author: leo
"""

import json
import requests

import urllib2

protocol = 'http://'
host = '127.0.0.1:5000/'


# List projects
to_append = 'admin/list_projects'
url = protocol + host + to_append
resp = requests.get(url)

if resp.ok:
    parsed_resp = json.loads(resp.content)
else: 
    print('Problem with list projects')
    
# Select id
_id = '5a48099d7611bcd50b94e038f6ffb3b7'

# Get metadata
to_append = 'project/metadata/{0}'.format(_id)
url = protocol + host + to_append
resp = requests.post(url)

if resp.ok:
    parsed_resp = json.loads(resp.content)
else: 
    print('Problem with metadata')

# Infer Missing values
to_append = 'project/modules/infer_mvs/{0}'.format(_id)
url = protocol + host + to_append
resp = requests.post(url)

if resp.ok:
    parsed_resp = json.loads(resp.content)
else: 
    print('Problem with infer_mvs')
    
# Replace Missing Values
to_append = 'project/modules/replace_mvs/{0}'.format(_id)
url = protocol + host + to_append
body = {'params': parsed_resp['response']}
resp = requests.post(url, json=body)   

if resp.ok:
    parsed_resp = json.loads(resp.content)
else: 
    print('Problem with replace_mvs')
    
    
print resp.content