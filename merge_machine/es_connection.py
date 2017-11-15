#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:48:01 2017

@author: m75380
"""
from elasticsearch import Elasticsearch

es = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)