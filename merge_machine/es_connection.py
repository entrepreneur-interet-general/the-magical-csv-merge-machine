#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:48:01 2017

@author: m75380
"""
from distutils.version import LooseVersion

import time

from elasticsearch import Elasticsearch, client

import CONFIG
if CONFIG.PRODUCTION_MODE:
    for i in range(0, 3): # Max retries
        while True:
            try:
                es = Elasticsearch('http://elasticsearch:9200', http_auth=('elastic', 'changeme'),
                    timeout=60, max_retries=10, retry_on_timeout=True)
                if not es.ping():
                    raise ValueError("ElasticSearch connection failed!")
            except:
                time.sleep(30)
                continue
            break
else:
    es = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)
es_version = es.info()['version']['number']

ic = client.IndicesClient(es)

if LooseVersion(es_version) < LooseVersion('5.6.1'):
    raise RuntimeError('ES Version is too old. Upgrade to 5.6.1 or newer.')
