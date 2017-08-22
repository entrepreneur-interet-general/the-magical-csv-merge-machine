#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:42:41 2017

@author: m75380

# Ideas: Learn analysers and weights for blocking on ES directly
# Put all fields to learn blocking by exact match on other fields

https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html
"""
 
import json
import os
import time

from elasticsearch import Elasticsearch, client
import pandas as pd

import pprint
import unidecode


dir_path = 'data/sirene'
chunksize = 20000
file_len = 10*10**6


test_num = 2
if test_num == 0:
    source_file_path = 'local_test_data/source.csv'
    match_cols = [{'source': 'commune', 'ref': 'L6_NORMALISEE'},
                  {'source': 'lycees_sources', 'ref': 'NOMEN_LONG'}]    
    source_sep = ','
    source_encoding = 'utf-8'
    
elif test_num == 1:
    source_file_path = 'local_test_data/integration_5/data_ugly.csv'
    match_cols = [{'source': 'VILLE', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ETABLISSEMENT', 'ref': 'NOMEN_LONG'}]
    source_sep = ';'
    source_encoding = 'windows-1252'
    
elif test_num == 2:
    source_file_path = 'local_test_data/integration_3/export_alimconfiance.csv'
    match_cols = [{'source': 'Code_postal', 'ref': 'LIBCOM'},
                  #{'source': 'Libelle_commune', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ods_adresse', 'ref': 'L4_NORMALISEE'},
                  {'source': 'APP_Libelle_etablissement', 'ref': 'L1_NORMALISEE', 'boost': 3}]
    source_sep = ';'
    source_encoding = 'utf-8'
else:
    raise Exception('Not a valid test number')


#



source = pd.read_csv(source_file_path, 
                    sep=source_sep, encoding=source_encoding,
                    dtype=str, nrows=chunksize)

ref_file_name = 'sirc-17804_9075_14209_201612_L_M_20170104_171522721.csv' # 'petit_sirene.csv'
ref_sep = ';'
ref_encoding = 'windows-1252'


columns_to_index = ['SIREN', 'NIC', 'L1_NORMALISEE', 'L2_NORMALISEE', 'L3_NORMALISEE',
       'L4_NORMALISEE', 'L5_NORMALISEE', 'L6_NORMALISEE', 'L7_NORMALISEE',
       'L1_DECLAREE', 'L2_DECLAREE', 'L3_DECLAREE', 'L4_DECLAREE',
       'L5_DECLAREE', 'L6_DECLAREE', 'L7_DECLAREE', 'LIBCOM', 'CEDEX', 'ENSEIGNE', 'NOMEN_LONG']

# default is 'keyword
columns_to_index = {
                    'SIREN': [], 
                    'NIC': [],
                    'L1_NORMALISEE': ['french', 'whitespace', 'integers'],
                    'L4_NORMALISEE': ['french', 'whitespace', 'integers'], 
                    'L6_NORMALISEE': ['french', 'whitespace', 'integers'],
                    'L1_DECLAREE': ['french', 'whitespace', 'integers'], 
                    'L4_DECLAREE': ['french', 'whitespace', 'integers'],
                    'L6_DECLAREE': ['french', 'whitespace', 'integers'],
                    'LIBCOM': ['french', 'whitespace', 'integers'], 
                    'CEDEX': [], 
                    'ENSEIGNE': ['french', 'whitespace', 'integers'], 
                    'NOMEN_LONG': ['french', 'whitespace', 'integers'],
                    # Keyword only
                    'LIBNATETAB': [],
                    'LIBAPET': [],
                    'PRODEN': [],
                    'PRODET': []
                    }

ref_gen = pd.read_csv(os.path.join('local_test_data', 'sirene', ref_file_name), 
                  sep=ref_sep, encoding=ref_encoding,
                  usecols=columns_to_index.keys(),
                  dtype=str, chunksize=chunksize, nrows=10**50)




#==============================================================================
# Index in Elasticsearch and test
#==============================================================================
testing = True

if testing:
    table_name = '123vivalalgerie2'
else:
    table_name = '123vivalalgerie'

es = Elasticsearch()


index_settings = {
                     "settings": {
                            "analysis": {
                                 "tokenizer" : {
                                    "integers" : {
                                       "type" : "pattern",
                                       "preserve_original" : 0,
                                       "pattern" : '(\\d+)',
                                       'group': 1
                                    }
                                 },                          
                                    
                                    
                                  "analyzer": {
                                    "integers": {
                                            'tokenizer': 'integers'
                                    }
                              }
                            }
                          },
                
                    "mappings": {
                            "structure": {                      
                            }
                          }
                }


field_mappings = {key: {'analyzer': 'keyword', 
                        'type': 'string',
                        'fields': {analyzer: {'type': 'string', 'analyzer': analyzer} 
                                    for analyzer in values}
                        } 
                for key, values in columns_to_index.items() if values}
field_mappings.update({key: {'analyzer': 'keyword', 
                             'type': 'string'
                        } 
                for key, values in columns_to_index.items() if not values})
                        
index_settings['mappings']['structure']['properties'] = field_mappings

new_index = False
do_indexing = False

if new_index:
    ic = client.IndicesClient(es)
    if ic.exists(table_name):
        ic.delete(table_name)
    ic.create(table_name, body=json.dumps(index_settings))  
    

# ic.put_settings



def pre_process(tab):
    '''     '''
    for x in tab.columns:
        tab.loc[:, x] = tab[x].str.strip()
    return tab

if do_indexing:    
    # For efficiency, reset refresh interval
    # see https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-update-settings.html
    if not testing:
        low_refresh = {"index" : {"refresh_interval" : "-1"}}
        ic.put_settings(low_refresh, table_name)
    
    # Bulk insert
    print('Started indexing')    
    i = 0
    a = time.time()
    for ref_tab in ref_gen:
        ref_tab = pre_process(ref_tab)
        body = ''
        for key, doc in ref_tab.where(ref_tab.notnull(), None).to_dict('index').items():
            index_order = json.dumps({
                                "index": {
                                          "_index": table_name, 
                                          "_type": 'structure', 
                                          "_id": str(key)
                                         }
                                })
            body += index_order + '\n'
            body += json.dumps(doc) + '\n'
        res = es.bulk(body)
        i += len(ref_tab)
        
        # Display progress
        b = time.time()
        eta = (file_len - i) * (b-a) / i
        print('Indexed {0} rows / ETA: {1} s'.format(i, eta))

    # Back to default refresh
    if not testing:
        default_refresh = {"index" : {"refresh_interval" : "1s"}}
        ic.put_settings(default_refresh, table_name)
        
    # TODO: what is this for ?
    es.indices.refresh(index=table_name)

def my_unidecode(string):
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''

'''
Methodology:
    
    
TABLES:
    - { source_hash: source_doc }
    - { pair_id : {'match': is_match, 'source': source_hash, 'ref': doc_id ...}
    - { query_template : {pair_id: (doc_id, score) ...} ...}

Evaluate possibilities of refining score according to number of potential candidates

1) After a few source labels
               
At each first labels 
1) Update requests in table: request_hash -> {(doc_id, score)...}
2) Update query templates: query_template_id -> [}]
3) Compute score for each predicate/query template.
4) Gradually increase number of disctinct values for normalized weight

Hash the normalized weight vectors to avoid computing both [2,1] and [4,2]
                            
                            
After 3 matches: discard those with worst precision


'''
        
query_template = {}
        
    

a = []
good = []
for i in range(100, 200):
    row = source.iloc[i]            
    
    to_ask = {x['ref']: {'query': my_unidecode(row[x['source']]), 
                        'boost': x.get('boost', 1)} \
                        for x in match_cols}
    
    body = {
          'query': {
            'bool': {
              'should': [
                    {'match': {key: {'query': val['query'],
                                     'boost': val['boost']}}} \
                                for key, val in to_ask.items() if val['boost']
                        ]
                    }
                  }
        }
    
    res = es.search(index=table_name, body=json.dumps(body), explain=False)
    
    #    may_have_match = 'lycee' in res.__str__().lower()
    #    print(may_have_match)
    #    if not may_have_match:
    #        print(to_ask)
    #    a.append(may_have_match)
    
    best_res = res['hits']['hits'][0]['_source']
    
    print('\n****')
    
    for key in sorted(to_ask.keys()):
        print(key, ' : ', to_ask[key]['query'])
        print(' '*len(key), '-> ', best_res[key])
    
    print('(', i, ') -> ', res['hits']['hits'][0]['_score'])
    
    is_good = input('Is good?\n >')
    good.append(is_good in {'1', 'y'})
    
    
# test_bulk_search
to_ask = {'L1_DECLAREE.french': {'boost': 4.5, 'query': 'ECOLe'}}
body = {
      'query': {
        'bool': {
          'should': [
                {'match': {key: {'query': val['query'],
                                 'boost': val['boost']}}} \
                            for key, val in to_ask.items() if val['boost']
                    ]
                }
              }
    }
          
bulk_body = ''
for x in range(10000):
    bulk_body += json.dumps({"index" : table_name}) + '\n'
    body = {'query': {'bool': {'should': [{'match': {'L1_DECLAREE.french': {'boost': 4.5,
       'query': str(x)}}}]}}}
    bulk_body += json.dumps(body) + '\n'
res = es.msearch(bulk_body) #, index=table_name)

print(sum('error' in x for x in res['responses']))
print(sum(bool(x['hits']['hits']) for x in res['responses'] if 'error' not in x))