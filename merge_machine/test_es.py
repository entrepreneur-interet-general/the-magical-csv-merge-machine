#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:42:41 2017

@author: m75380

# Ideas: Learn analysers and weights for blocking on ES directly
# Put all fields to learn blocking by exact match on other fields

https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html
"""
import itertools
import json
import os
import time

from elasticsearch import Elasticsearch, client
import numpy as np
import pandas as pd

import pprint
import unidecode


dir_path = 'data/sirene'
chunksize = 3000
file_len = 10*10**6


test_num = 2
if test_num == 0:
    source_file_path = 'local_test_data/source.csv'
    match_cols = [{'source': 'commune', 'ref': 'LIBCOM'},
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
#    match_cols = [{'source': 'Libelle_commune', 'ref': 'LIBCOM'},
#                  #{'source': 'Libelle_commune', 'ref': 'L6_NORMALISEE'},
#                  {'source': 'ods_adresse', 'ref': 'L4_NORMALISEE'},
#                  {'source': 'APP_Libelle_etablissement', 'ref': 'L1_NORMALISEE'},
#                  {'source': 'APP_Libelle_etablissement', 'ref': 'ENSEIGNE'},
#                  {'source': 'APP_Libelle_etablissement', 'ref': 'NOMEN_LONG'}]
    match_cols = [{'source': 'Libelle_commune', 'ref': 'LIBCOM'},
                  #{'source': 'Libelle_commune', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ods_adresse', 'ref': 'L4_NORMALISEE'},
                  {'source': 'APP_Libelle_etablissement', 'ref': ('L1_NORMALISEE', 
                                                        'ENSEIGNE', 'NOMEN_LONG')}]

    source_sep = ';'
    source_encoding = 'utf-8'
else:
    raise Exception('Not a valid test number')


#



source = pd.read_csv(source_file_path, 
                    sep=source_sep, encoding=source_encoding,
                    dtype=str, nrows=chunksize)
source = source.where(source.notnull(), '')


ref_file_name = 'sirc-17804_9075_14209_201612_L_M_20170104_171522721.csv' # 'petit_sirene.csv'
# ref_file_name = 'petit_sirene.csv'
ref_sep = ';'
ref_encoding = 'windows-1252'


#columns_to_index = ['SIREN', 'NIC', 'L1_NORMALISEE', 'L2_NORMALISEE', 'L3_NORMALISEE',
#       'L4_NORMALISEE', 'L5_NORMALISEE', 'L6_NORMALISEE', 'L7_NORMALISEE',
#       'L1_DECLAREE', 'L2_DECLAREE', 'L3_DECLAREE', 'L4_DECLAREE',
#       'L5_DECLAREE', 'L6_DECLAREE', 'L7_DECLAREE', 'LIBCOM', 'CEDEX', 'ENSEIGNE', 'NOMEN_LONG']

# default is 'keyword
columns_to_index = {
                    'SIREN': {}, 
                    'NIC': {},
                    'L1_NORMALISEE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'},
                    'L4_NORMALISEE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'}, 
                    'L6_NORMALISEE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'},
                    'L1_DECLAREE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'}, 
                    'L4_DECLAREE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'},
                    'L6_DECLAREE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'},
                    'LIBCOM': {'french', 'whitespace', 'end_n_grams', 'n_grams'}, 
                    'CEDEX': {}, 
                    'ENSEIGNE': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'}, 
                    'NOMEN_LONG': {'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'},
                    # Keyword only
                    'LIBNATETAB': {},
                    'LIBAPET': {},
                    'PRODEN': {},
                    'PRODET': {}
                    }

ref_gen = pd.read_csv(os.path.join('local_test_data', 'sirene', ref_file_name), 
                  sep=ref_sep, encoding=ref_encoding,
                  usecols=columns_to_index.keys(),
                  dtype=str, chunksize=chunksize, nrows=10**50) 

#==============================================================================
# Index in Elasticsearch and test
#==============================================================================
testing = True
new_index = False
do_indexing = False


if testing:
    table_name = '123vivalalgerie'
else:
    table_name = '123vivalalgerie3'

es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)

# https://www.elastic.co/guide/en/elasticsearch/reference/1.4/analysis-edgengram-tokenizer.html

tokenizers = {
                "integers" : {
                               "type" : "pattern",
                               "preserve_original" : 0,
                               "pattern" : '(\\d+)',
                               'group': 1
                               },
                "n_grams": {
                              "type": "ngram",
                              "min_gram": 3,
                              "max_gram": 3,
                              "token_chars": [
                                "letter",
                                "digit"
                              ]
                            }
                }

filters =  {
                "my_edgeNGram": {
                    "type":       "edgeNGram",
                    "min_gram": 3,
                    "max_gram": 30
            }}

analyzers = {
                "integers": {'tokenizer': 'integers'},
                "n_grams": {'tokenizer': 'n_grams'},
                "end_n_grams": {'tokenizer': 'keyword',
                                "filter" : ["reverse", "my_edgeNGram", "reverse"]}
            }


index_settings = {
                     "settings": {
                            "analysis": {
                                 "tokenizer": tokenizers,
                                 "filter": filters,
                                 "analyzer": analyzers
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


if new_index:
    ic = client.IndicesClient(es)
    if ic.exists(table_name):
        ic.delete(table_name)
    ic.create(table_name, body=json.dumps(index_settings))  


def pre_process(tab):
    '''     '''
    for x in tab.columns:
        tab.loc[:, x] = tab[x].str.strip()
    return tab

def index(ref_gen, testing):
    '''Index ref_gen'''
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
        # TODO: REMOVE THIS
        if ref_tab.index[-1] <= int(7.5*10**6):
            continue
        
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
        _ = es.bulk(body)
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

if do_indexing:
    index(ref_gen, testing)

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

def compute_metrics(hits, ref_id):
    '''Computes metrics for hits: res['hits']['hits']'''
    metrics = dict()
    
    # General query metrics
    num_hits = len(hits)

    if num_hits:
        _score_first = hits[0]['_score']
    else:
        _score_first = 0
    
    # Match metrics
    try:
        i = next((i for i, hit in enumerate(hits) if hit['_id']==ref_id))
        has_match = True
    except StopIteration:
        has_match = False
        
    if has_match:
        is_first = i == 0        
        es_id_score = hits[i]['_score']
        pos_score = 1 / (1+i)
    else:
        is_first = False
        es_id_score = 0
        pos_score = 0
        
    metrics['_score_first'] = _score_first
    metrics['num_hits'] = num_hits
    metrics['has_match'] = has_match
    metrics['is_first'] = is_first
    metrics['es_id_score'] = es_id_score
    metrics['pos_score'] = pos_score
    
    return metrics

def _gen_suffix(columns_to_index, s_q_t_2):
    '''Yields suffixes to add to field_names for the given analyzers'''
    if isinstance(s_q_t_2, str):
        analyzers = columns_to_index[s_q_t_2]
    elif isinstance(s_q_t_2, tuple):
        analyzers = set.union(*[columns_to_index[col] for col in s_q_t_2])
    yield '' # No suffix for standard analyzer
    for analyzer in analyzers:
        yield '.' + analyzer


# TODO: bool levels for fields also (L4 for example)

# query_template is ((source_key, ref_key, analyzer_suffix, boost), ...)

max_num_levels = 3 # Number of match clauses
bool_levels = {'.integers': ['must', 'should']}
#len(query_metrics[list(query_metrics.keys())[0]])
boost_levels = [1]
single_queries = list(((bool_lvl, x['source'], x['ref'], suffix, boost) \
                                   for x in match_cols \
                                   for suffix in _gen_suffix(columns_to_index, x['ref']) \
                                   for bool_lvl in bool_levels.get(suffix, ['must']) \
                                   for boost in boost_levels))
all_query_templates = list(itertools.chain(*[list(itertools.combinations(single_queries, x)) \
                                    for x in range(2, max_num_levels+1)][::-1]))
# Queries must contain all columns at least two distinct columns
if len(match_cols) >= 1:
    all_query_templates = list(filter(lambda query: len(set((x[1], x[2]) for x in query)) >= 2, \
                                all_query_templates))

#all_query_templates = [((y, 'ods_adresse', 'L4_NORMALISEE', '.n_grams', 1), 
#                        ('must', 'APP_Libelle_etablissement', 'NOMEN_LONG', '', 1)) \
#                    for y in ['should', 'must']] \
#                + [((y, 'ods_adresse', 'L4_NORMALISEE', '.n_grams', 1), 
#                  (z, 'APP_Libelle_etablissement', 'NOMEN_LONG', '.french', x), 
#                  (v, 'APP_Libelle_etablissement', 'NOMEN_LONG', '.french', x2)) \
#                for x in range(1,4) for x2 in range(1,4) \
#                for y in ['should', 'must'] \
#                for z in ['should', 'must'] \
#                for v in ['should', 'must']]

def _gen_body(query_template, row, num_results=3):
    '''
    query_template is ((source_col, ref_col, analyzer_suffix, boost), )
    row is pandas.Series from the source object
    
    s_q_t: single_query_template
    '''
    #    source_val = row[s_q_t[1]]
    #    key = s_q_t[2] + s_q_t[3]
    #    boost = s_q_t[4]
    
    body = {
          'size': num_results,
          'query': {
            'bool': {
               must_or_should: [
                          {'match': {
                                  s_q_t[2] + s_q_t[3]: {'query': my_unidecode(row[s_q_t[1]].lower()).replace('lycee', ''),
                                                        'boost': s_q_t[4]}}
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and isinstance(s_q_t[2], str)
                        ] \
                        + [
                          {'multi_match': {
                                  'fields': [col + s_q_t[3] for col in s_q_t[2]], 
                                  'query': my_unidecode(row[s_q_t[1]].lower()).replace('lycee', ''),
                                  'boost': s_q_t[4]
                                  }
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and isinstance(s_q_t[2], tuple)
                        ] \
                 for must_or_should in ['must', 'should']   
                    #,
#               'filter': [{'match': {'NOMEN_LONG.french': {'query': 'Lycee'}}
#                         }],
#               'must_not': [{'match': {'NOMEN_LONG.french': {'query': 'amicale du OR ass or association OR foyer OR sportive OR parents OR MAISON DES'}}
#                         }]
                    }
                  }
           }
    return body

def compute_threshold(metrics, t_p=0.99, t_r=0.3):
    ''' 
    Compute the optimale threshold and the metrics associated
    
    t_p = 0.9 # target_precision
    t_r = 0.3 # target_recall    
    '''
    num_metrics = len(metrics)

    # Scores deal with empty hits
    sorted_metrics = sorted(metrics, key=lambda x: x['_score_first'], reverse=True)
    
    # score_vect = np.array([x['_score_first'] for x in sorted_metrics])
    is_first_vect = np.array([bool(x['is_first']) for x in sorted_metrics])
    # rolling_score_mean = score_vect.cumsum() / (np.arange(num_metrics) + 1)
    rolling_precision = is_first_vect.cumsum() / (np.arange(num_metrics) + 1)
    rolling_recall = is_first_vect.cumsum() / num_metrics
    
    # WHY is recall same as precision ?
    
    # Compute ratio
    _f_precision = lambda x: max(x - t_p, 0) + min(t_p*(x/t_p)**3, t_p)
    _f_recall = lambda x: max(x - t_r, 0) + min(t_p*(x/t_r)**3, t_r)
    a = np.fromiter((_f_precision(xi) for xi in rolling_precision), rolling_precision.dtype)
    b = np.fromiter((_f_recall(xi) for xi in rolling_recall), rolling_recall.dtype)
    rolling_ratio = a*b

    # Find best index for threshold
    idx = max(num_metrics - rolling_ratio[::-1].argmax() - 1, min(6, num_metrics-1))
    
    if sum(is_first_vect) == 0:
        return 10**3, 0, 0, 0
    else:
        return sorted_metrics[idx]['_score_first'], rolling_precision[idx], rolling_recall[idx], rolling_ratio[idx]


def calc_agg_query_metrics(query_metrics):
    agg_query_metrics = dict()
    for key, metrics in query_metrics.items():
        thresh, precision, recall, ratio = compute_threshold(metrics)
        agg_query_metrics[key] = dict()
        try:
            agg_query_metrics[key]['precision'] = sum(x['is_first'] and (x['_score_first'] >= thresh) for x in metrics) \
                                                / (sum(bool(x['num_hits'])  and (x['_score_first'] >= thresh) for x in metrics) or 1)
        except:
            import pdb; pdb.set_trace()
        agg_query_metrics[key]['recall'] = sum(x['is_first'] and (x['_score_first'] >= thresh) for x in metrics) / len(metrics)
        
    return agg_query_metrics

    
    
def find_match(full_responses, sorted_keys, row, num_results, num_rows_labelled):
    '''
    User labelling going through potential results (order given by sorted_keys) looking for a 
    match
    
    OUTPUT:
        - found: If a match was found
        - res: result of the match if it was found
    '''
    ids_done = []
    for key in sorted_keys:
        print('\nkey: ', key)
        results = full_responses[key]['hits']['hits']
        for res in results[:num_results]:
            if res['_id'] not in ids_done and ((res['_score']>=0.001)):
                ids_done.append(res['_id'])
                print('\n***** {0} / {1} / ({2})'.format(res['_id'], res['_score'], num_rows_labelled))
                for match in match_cols:
                    if isinstance(match['ref'], str):
                        cols = [match['ref']]
                    else:
                        cols = match['ref']
                    for col in cols:
                        print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
                        print('> {1}   -> [{0}]'.format(col, res['_source'][col]))
                
                if test_num == 2:
                    print(row['SIRET'][:-5], row['SIRET'][-5:], '[source]')
                    print(res['_source']['SIREN'], res['_source']['NIC'], '[ref]')
                    print('LIBAPET', res['_source']['LIBAPET'])
                    is_match = row['SIRET'] == res['_source']['SIREN'] + res['_source']['NIC']
                else:
                    is_match = input('Is match?\n > ') in ['1', 'y']
                if is_match:
                    return True, res
                else:
                    print('not first')
    return False, None    

def make_bulk(search_templates, rows, num_results, chunk_size=100):
    queries = []
    bulk_body = ''
    i = 0
    for (q_t, row) in search_templates:
        bulk_body += json.dumps({"index" : table_name}) + '\n'
        body = _gen_body(q_t, row, num_results)
        bulk_body += json.dumps(body) + '\n'
        queries.append((q_t, row))
        i += 1
        if i == chunk_size:
            yield bulk_body, queries
            queries = []
            bulk_body = ''
            i = 0
    yield bulk_body, queries

def perform_queries(all_query_templates, rows, num_results=3):
    '''
    Searches for the values in row with all the search templates in all_query_templates.
    Retry on error
    '''
    i = 1
    full_responses = dict() 
    og_search_templates = list(enumerate(itertools.product(all_query_templates, rows)))
    search_templates = og_search_templates
    # search_template is [(id, (query, row)), ...]
    while search_templates:
        print('At search iteration', i)
        
        bulk_body_gen = make_bulk([x[1] for x in search_templates], rows, num_results)
        responses = []
        for bulk_body, _ in bulk_body_gen:
            responses += es.msearch(bulk_body)['responses'] #, index=table_name)
        
        has_error_vect = ['error' in x for x in responses]
        has_hits_vect = [('error' not in x) and bool(x['hits']['hits']) for x in responses]
        
        # Update for valid responses
        for (s_t, res, has_error) in zip(search_templates, responses, has_error_vect):
            if not has_error:
                full_responses[s_t[0]] = res
        
        print('Num errors:', sum(has_error_vect))
        print('Num hits', sum(has_hits_vect))
        
        # Limit query to those we couldn't get the first time
        search_templates = [x for x, y in zip(search_templates, has_error_vect) if y]
        i += 1
        
        if i >= 10:
            raise Exception('Problem with elasticsearch')
            
    return og_search_templates, full_responses

def match(source, query_template, threshold):
    '''Return concatenation of source and reference with the matches found'''
    rows = (x[1] for x in source.iterrows())
    all_search_templates, full_responses = perform_queries([query_template], rows, num_results=1)
    full_responses = [full_responses[i] for i in range(len(full_responses))] # Don't use items to preserve order
    
    matches_in_ref = pd.DataFrame([f_r['hits']['hits'][0]['_source'] \
                               if bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold) \
                               else {} \
                               for f_r in full_responses])
                    
    confidence = pd.Series([f_r['hits']['hits'][0]['_score'] \
                            if bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold) \
                            else np.nan \
                            for f_r in full_responses])
    
    new_source = pd.concat([source, matches_in_ref], 1)
    new_source['__CONFIDENCE'] = confidence
    return new_source



query_metrics = dict()
for q_t in all_query_templates:
    query_metrics[q_t] = []

import random

num_results_labelling = 3

labels = []

num_found = 0
if test_num == 0:
    num_samples = 0
else:
    num_samples = 100
for num_rows_labelled, idx in enumerate(random.sample(list(source.index), num_samples)):
    row = source.loc[idx]
    print('Doing', row)  
        
    # Search elasticsearch for all results
    all_search_templates, full_responses = perform_queries(all_query_templates, [row], num_results_labelling)
    full_responses = {all_search_templates[idx][1][0]: values for idx, values in full_responses.items()}
    
    # Sort keys by score or most promising
    use_precision = num_found > 2
    if not use_precision:
        sorted_keys = random.sample(list(full_responses.keys()), len(full_responses.keys()))
        # sorted_keys = sorted(full_responses.keys(), key=lambda x: full_responses[x]['hits'].get('max_score') or 0, reverse=True)
    else:
        print('Using best precision')
        if found:
            agg_query_metrics = calc_agg_query_metrics(query_metrics)
        sorted_keys = sorted(full_responses.keys(), key=lambda x: agg_query_metrics[x]['precision'], reverse=True)
        [agg_query_metrics[x] for x in sorted_keys[:10]]
    # Try to find the match somewhere


    found, res = find_match(full_responses, sorted_keys, row, num_results_labelling, num_rows_labelled)

    if found:
        num_found += 1
        for key, response in full_responses.items():
            query_metrics[key].append(compute_metrics(response['hits']['hits'], res['_id']))



        #        import pdb
        #        pdb.set_trace()
        
# TODO: thresholder

# Sort by max score?

# Propose by descending score ?
# Alternate between my precision score and ES max score (or combination)

# TODO: only on chunksize rows of source

if test_num == 0:
    best_query_template = (('must', 'commune', 'LIBCOM', '.french', 1),
     ('must', 'lycees_sources', 'NOMEN_LONG', '.french', 10))
else:
    best_query_template = sorted_keys[0]

new_source = match(source, best_query_template, 5)
new_source['has_match'] = new_source.__CONFIDENCE.notnull()


# Display results
new_source.sort_values('__CONFIDENCE', inplace=True)
for i, row in new_source[new_source.has_match].iloc[:100].iterrows(): 
    print('**** ({0}) / score: {1}\n'.format(i, row['__CONFIDENCE']))
    for match in match_cols:
        print(row[match['source']], '\n', row[match['ref']], '\n')

assert False

# Number of results
len(query_metrics[list(query_metrics.keys())[0]])

# Sorted metrics
for x in sorted_keys[:40]:
    print(x, '\n -> ', agg_query_metrics[x], '\n')

t_ps = [x/200. for x in range(2*80, 2*100)]
precs = []
recs = []

for t_p in t_ps:
    tab = pd.DataFrame([(key, *compute_threshold(metrics, t_p=t_p)) for key, metrics in query_metrics.items()])
    tab.columns = ['key', 'thresh', 'precision', 'recall', 'ratio']
    tab.sort_values('ratio', ascending=False, inplace=True)
    precs.append(tab.precision.iloc[0])
    recs.append(tab.recall.iloc[0])
    
import matplotlib.pyplot as plt
plt.plot(t_ps, precs)
plt.plot(t_ps, recs)
print(tab.head(10))

'''
pre-process:
COMMUNE: remove "commune de"

pre-process: L1 + ENSEIGNE + NOMEN_LONG

Redresser les noms propres A. Camuse

tags pour les LIBAPET sirene ?

'''


'''
DATA?
Rungis   -> [Libelle_commune][source]
> CHEVILLY LARUE   -> [LIBCOM]
BAT E4 DU MIN DE PARIS RUNGIS   -> [ods_adresse][source]
> 2 AVENUE DE FLANDRE   -> [L4_NORMALISEE]
PALIMEX   -> [APP_Libelle_etablissement][source]
> PALIMEX   -> [L1_NORMALISEE]
330177635 00050 [source]
330177635 00050 [ref]

Marseille 7e  Arrondissement   -> [Libelle_commune][source]
> CHALLES LES EAUX   -> [LIBCOM]
LA LICORNE - PIZZERIA   -> [ods_adresse][source]
> 72 ALL DE LA BREISSE - RESIDENCE   -> [L4_NORMALISEE]
LA LICORNE   -> [APP_Libelle_etablissement][source]
> LGDI   -> [L1_NORMALISEE]
LA LICORNE   -> [APP_Libelle_etablissement][source]
> None   -> [ENSEIGNE]
822461620 00012 [source]
823997358 00010 [ref]

sans paramêtrage: Lycée Maxence Van der Meersch 
 ASS MAXENCE VAN DER MEERSCH 

Paris 14e  Arrondissement   -> [Libelle_commune][source]
> PARIS 14   -> [LIBCOM]

181 RUE DU CHATEAU   -> [ods_adresse][source]
> 171 RUE DU CHATEAU   -> [L4_NORMALISEE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> L ASSIETTE   -> [L1_NORMALISEE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> L'ASSIETTE   -> [ENSEIGNE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> NUAGE SARL   -> [NOMEN_LONG]
505240614 00014 [source]
505240614 00014 [ref]

Take queries with better precision first

'''
