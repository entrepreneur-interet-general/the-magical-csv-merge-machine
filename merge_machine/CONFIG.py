#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:41:34 2017

@author: leo
"""

import os 
import sys

PRODUCTION_MODE = True

cwd = os.getcwd()

if not PRODUCTION_MODE:
	cwd = os.path.split((os.path.join(cwd, sys.argv[0])))[0]

DATA_PATH = os.path.join(cwd, 'data')
LINK_DATA_PATH = os.path.join(cwd, 'data/link')
NORMALIZE_DATA_PATH = os.path.join(cwd, 'data/normalize')
RESOURCE_PATH = os.path.join(cwd, 'resource')

print('DATA_PATH\n', DATA_PATH)
print('LINK_DATA_PATH\n', LINK_DATA_PATH)
print('NORMALIZE_DATA_PATH\n', NORMALIZE_DATA_PATH)
print('RESOURCE_PATH\n', RESOURCE_PATH)