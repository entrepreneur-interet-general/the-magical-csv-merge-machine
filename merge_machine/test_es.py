#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:42:41 2017

@author: m75380

# Ideas: Learn analysers and weights for blocking on ES directly
# Put all fields to learn blocking by exact match on other fields

https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html

$ ./bin/elasticsearch
    

        queries = self.current_queries
        
        new_res = dict()
        source_idxs = self.current_queries[0].history_pairs.keys()
        for idx in source_idxs:
            count = defaultdict(int)
            for query in queries:
                if query.history_pairs[idx]:
                    count[query.history_pairs[idx][0]] += 1
                else:
                    count['nores'] += 1
            new_res[idx] = sorted(list(count.items()), key=lambda x: x[1], reverse=True)[0][0]

"""

from elasticsearch import Elasticsearch, client
import pandas as pd

from es_connection import es
from es_labeller import ConsoleLabeller


dir_path = 'data/sirene'
chunksize = 3000
file_len = 10*10**6


force_re_index = False

sirene_index_name = '123vivalalgerie2'

test_num = 0

if test_num == 0:
    source_file_path = 'local_test_data/source.csv'
    ref_file_path = ''
    match_cols = [{'source': 'commune', 'ref': 'LIBCOM'},
                  {'source': 'lycees_sources', 'ref': 'NOMEN_LONG'}]    
    source_sep = ','
    source_encoding = 'utf-8'
    
    
    ref_table_name = sirene_index_name
    
elif test_num == 1:
    source_file_path = 'local_test_data/integration_5/data_ugly.csv'
    match_cols = [{'source': 'VILLE', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ETABLISSEMENT', 'ref': 'NOMEN_LONG'}]
    source_sep = ';'
    source_encoding = 'windows-1252'
    
    ref_table_name = sirene_index_name
    
elif test_num == 2:
    # ALIM to SIRENE
    source_file_path = 'local_test_data/integration_3/export_alimconfiance.csv'
    match_cols = [{'source': 'Libelle_commune', 'ref': 'LIBCOM'},
                  #{'source': 'Libelle_commune', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ods_adresse', 'ref': 'L4_NORMALISEE'},
                  {'source': 'APP_Libelle_etablissement', 'ref': ('L1_NORMALISEE', 
                                                        'ENSEIGNE', 'NOMEN_LONG')}]

    source_sep = ';'
    source_encoding = 'utf-8'
    
    ref_table_name = sirene_index_name
    
elif test_num == 3:
    # HAL to GRID
    source_file_path = 'local_test_data/integration_4/hal.csv'

    match_cols = [{
                    "source": ("parentName_s", "label_s"),
                    "ref": ("Name", "City")
                  }]
    source_sep = '\t'
    source_encoding = 'utf-8'
    
    ref_table_name = '01c670508e478300b9ab7c639a76c871'

elif test_num == 4:
    source_file_path = 'local_test_data/integration_6_hal_2/2017_09_15_HAL_09_08_2015_Avec_RecageEchantillon.csv'

    match_cols = [{
                    "source": ("parentName_s", "label_s"),
                    "ref": ("Name", "City")
                  }]
    source_sep = ';'
    source_encoding = 'ISO-8859-1'
    
    ref_table_name = '01c670508e478300b9ab7c639a76c871'

elif test_num == 5:
    # Test on very short file
    source_file_path = 'local_test_data/source_5_lines.csv'
    match_cols = [{'source': 'commune', 'ref': 'LIBCOM'},
                  {'source': 'lycees_sources', 'ref': 'NOMEN_LONG'}]    
    source_sep = ','
    source_encoding = 'utf-8'
    
    ref_table_name = sirene_index_name    

else:
    raise Exception('Not a valid test number')
    
    
# =============================================================================
# Index the referential
# =============================================================================
    
testing = True     

# Initialize Elasticsearch connection
es = Elasticsearch(timeout=60, max_retries=10, retry_on_timeout=True)
ic = client.IndicesClient(es)

if force_re_index or ic.exists(ref_table_name):
    if ic.exists(ref_table_name):
        ic.delete(ref_table_name)
    
    
    

ref_gen = pd.read_csv(ref_path, 
                  usecols=columns_to_index.keys(),
                  dtype=str, chunksize=self.es_insert_chunksize)

if self.has_index() and force:
    self.ic.delete(self.index_name)
    
if not self.has_index():
    logging.info('Creating new index')
    log = self._init_active_log('INIT', 'transform')
    
    index_settings = es_insert.gen_index_settings(columns_to_index)
    
    logging.warning('Creating index')
    logging.warning(index_settings)
    self.ic.create(self.index_name, body=json.dumps(index_settings))    
    logging.warning('Inserting in index')
    es_insert.index(ref_gen, self.index_name, testing)

    log = self._end_active_log(log, error=False)
    
    

source = pd.read_csv(source_file_path, 
                    sep=source_sep, encoding=source_encoding,
                    dtype=str, nrows=chunksize)
source = source.where(source.notnull(), '')

if test_num in [0,1,2,5]:
    columns_to_index = {
        'SIRET': {},
        'SIREN': {},
        'NIC': {},
        'L1_NORMALISEE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'L4_NORMALISEE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'L6_NORMALISEE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'L1_DECLAREE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'L4_DECLAREE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'L6_DECLAREE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'LIBCOM': {
            'french', 'n_grams', 'city'
        },
        'CEDEX': {},
        'ENSEIGNE': {
            'french', 'integers', 'n_grams', 'city'
        },
        'NOMEN_LONG': {
            'french', 'integers', 'n_grams', 'city'
        },
        #Keyword only 'LIBNATETAB': {},
        'LIBAPET': {},
        'PRODEN': {},
        'PRODET': {}
    }
        
elif test_num in [3, 4]:
    columns_to_index = {
            "Name": {
                    'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams', 'city'
                    },
            "City": {
                    'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams', 'city'
                    }
            }

if test_num == 2:
    columns_certain_match = {'source': ['SIRET'], 'ref': ['SIRET']}
    labellers = dict()

    for i in range(3):
        labellers[i] = ConsoleLabeller(es, source, ref_table_name, match_cols, columns_to_index)
        labellers[i].auto_label(columns_certain_match)
        
    
#    import cProfile
#    cProfile.run("labeller.auto_label(columns_certain_match)", "restats")
#    
#    import pstats
#    p = pstats.Stats('restats')
#    p.strip_dirs().sort_stats(-1).print_stats()
    
elif test_num == 4:
    columns_certain_match = {'source': ['grid'], 'ref': ['ID']}
    labeller = ConsoleLabeller(es, source, ref_table_name, match_cols, columns_to_index)
    
else:    
    labeller = ConsoleLabeller(es, source, ref_table_name, match_cols, columns_to_index)


labeller.console_labeller()

new_source = es_linker(source, params)




#    if i == 15:
#        print('Updating musts')
#        if test_num == 0:
#            labeller.update_musts({'NOMEN_LONG': ['lycee']},
#                                  {'NOMEN_LONG': ['ass', 'association', 'sportive', 
#                                                  'foyer', 'maison', 'amicale']})


best_query = labeller.current_queries[0]
print(best_query._as_tuple())
print('Precision:', best_query.precision)
print('Recall:', best_query.recall)
print('Score:', best_query.score)

assert False

from collections import defaultdict
# Majority vote on labellers
pairs_count = dict()
for labeller in labellers.values():
    best_query = labeller.current_queries[0]
    
    for source_id, pairs in best_query.history_pairs.items():
        
        if source_id not in pairs_count:
            pairs_count[source_id] = defaultdict(int)
        
        if pairs:
            pair = pairs[0]
            pairs_count[source_id][pair] += 1
        
        
res = dict()
for source_id, pair_count in pairs_count.items():
    if pair_count:
        res[source_id] = sorted(list(pair_count.items()), key=lambda x: x[1])[-1][0]
    else:
        res[source_id] = None
    
