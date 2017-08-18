#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:42:41 2017

@author: m75380

# Ideas: Learn analysers and weights for blocking on ES directly
# Put all fields to learn blocking by exact match on other fields
"""
 
import json
import os
import time

from elasticsearch import Elasticsearch, client
import pandas as pd

import pprint
import unidecode


dir_path = 'data/sirene'
chunksize = 50000

#source_file_path = 'local_test_data/source.csv'
source_file_path = 'local_test_data/integration_5/data_ugly.csv'
source_sep = ';'
source_encoding = 'windows-1252'
source = pd.read_csv(source_file_path, 
                    sep=source_sep, encoding=source_encoding,
                    dtype=str, nrows=chunksize)

ref_file_name = 'sirc-17804_9075_14209_201612_L_M_20170104_171522721.csv' # 'petit_sirene.csv'
ref_sep = ';'
ref_encoding = 'windows-1252'


columns_to_index = ['SIREN', 'NIC', 'L1_NORMALISEE', 'L2_NORMALISEE', 'L3_NORMALISEE',
       'L4_NORMALISEE', 'L5_NORMALISEE', 'L6_NORMALISEE', 'L7_NORMALISEE',
       'L1_DECLAREE', 'L2_DECLAREE', 'L3_DECLAREE', 'L4_DECLAREE',
       'L5_DECLAREE', 'L6_DECLAREE', 'L7_DECLAREE', 'CEDEX', 'ENSEIGNE', 'NOMEN_LONG']

ref_gen = pd.read_csv(os.path.join('local_test_data', 'sirene', ref_file_name), 
                  sep=ref_sep, encoding=ref_encoding,
                  usecols=columns_to_index,
                  dtype=str, chunksize=10000, nrows=10**50)

#match_cols = [{'source': 'commune', 'ref': 'L6_NORMALISEE'},
#              {'source': 'lycees_sources', 'ref': 'NOMEN_LONG'}]

match_cols = [{'source': 'VILLE', 'ref': 'L6_NORMALISEE'},
              {'source': 'ETABLISSEMENT', 'ref': 'NOMEN_LONG'}]

#==============================================================================
# Index in Elasticsearch and test
#==============================================================================

table_name = '123vivalalgerie'

es = Elasticsearch()

delete_index = False
do_indexing = False

if do_indexing:
    if delete_index:
        ic = client.IndicesClient(es)
        ic.delete(table_name)
    #res = es.search(index=table_name, body={"query": {"match_all": {}}})
    
    # Bulk insert
    print('Started indexing')    
    i = 0
    for ref_tab in ref_gen:
        body = ''
        for key, doc in ref_tab.where(ref_tab.notnull(), None).to_dict('index').items():
            body += '{"index":{"_index":"' +  table_name + '", "_type":"' + 'structure' + '", "_id": ' + str(key) + '}}\n'
            body += json.dumps(doc) + '\n'
        res = es.bulk(body)
        i += len(ref_tab)
        print('Indexed {0} rows'.format(i))
        
    # TODO: what is this for ?
    es.indices.refresh(index=table_name)

def my_unidecode(string):
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''




a = []
good = []
for i in range(100, 200):
    row = source.iloc[i]            
    
    to_ask = {x['ref']: my_unidecode(row[x['source']]) for x in match_cols}
    
    body = {
          'query': {
            'bool': {
              'should': [
                { 'match': {key: value }} for key, value in to_ask.items()
              ]
            }
          }
        }
    
    res = es.search(index=table_name, body=json.dumps(body), explain=False)
    
    may_have_match = 'lycee' in res.__str__().lower()
    print(may_have_match)
    if not may_have_match:
        print(to_ask)
    a.append(may_have_match)
    
    best_res = res['hits']['hits'][0]['_source']
    
    print('\n****')
    pprint.pprint(to_ask)
    pprint.pprint({key: best_res[key] for key in to_ask.keys()})
    print('(', i, ') --> ', res['hits']['hits'][0]['_score'])
    
    is_good = input('Is good?\n >')
    good.append(is_good=='1')