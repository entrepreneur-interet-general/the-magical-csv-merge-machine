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

import unidecode


def pre_process_tab(tab):
    ''' Clean tab before insertion '''
    for x in tab.columns:
        tab.loc[:, x] = tab[x].str.strip()
    return tab

def index(ref_gen, table_name, testing=False):
    '''
    Insert values from ref_gen in the Elasticsearch index
    
    INPUT:
        - ref_gen: a pandas DataFrame generator (ref_gen=pd.read_csv(file_path, chunksize=XXX))
        - table_name: name of the Elasticsearch index
        - (testing): whether or not to refresh index at each insertion
    '''
    
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
        
        ref_tab = pre_process_tab(ref_tab)
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

def my_unidecode(string):
    '''unidecode or return empty string'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''


dir_path = 'data/sirene'
chunksize = 3000
file_len = 10*10**6


test_num = 1
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


# default is 'keyword

from es_config import columns_to_index, index_settings

ref_gen = pd.read_csv(os.path.join('local_test_data', 'sirene', ref_file_name), 
                  sep=ref_sep, encoding=ref_encoding,
                  usecols=columns_to_index.keys(),
                  dtype=str, chunksize=chunksize, nrows=10**50) 

#==============================================================================
# Index in Elasticsearch 
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


if new_index:
    ic = client.IndicesClient(es)
    if ic.exists(table_name):
        ic.delete(table_name)
    ic.create(table_name, body=json.dumps(index_settings))  

if do_indexing:
    index(ref_gen, table_name, testing)



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
    '''
    Computes metrics for based on the : res['hits']['hits']
    
    INPUT:
        - hits: res['hits']['hits']
        - ref_id: The real target Elasticsearch ID
        
    OUTPUT:
        - metrics: dict with various information    
    '''
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
    '''
    Yields suffixes to add to field_names for the given analyzers
    
    INPUT:
        - columns_to_index:
        - s_q_t_2: column or columns from ref
        
    NB: s_q_t_2: element two of the single_query_template
    
    '''
    if isinstance(s_q_t_2, str):
        analyzers = columns_to_index[s_q_t_2]
    elif isinstance(s_q_t_2, tuple):
        analyzers = set.union(*[columns_to_index[col] for col in s_q_t_2])
    yield '' # No suffix for standard analyzer
    for analyzer in analyzers:
        yield '.' + analyzer



def _gen_body(query_template, row, num_results=3):
    '''
    Generate the string to pass to Elastic search for it to execute query
    
    INPUT:
        - query_template: ((source_col, ref_col, analyzer_suffix, boost), ...)
        - row: pandas.Series from the source object
        - num_results: Max number of results for the query
    
    OUTPUT:
        - body: the query as string
    
    NB: s_q_t: single_query_template
        source_val = row[s_q_t[1]]
        key = s_q_t[2] + s_q_t[3]
        boost = s_q_t[4]
    '''

    
    body = {
          'size': num_results,
          'query': {
            'bool': dict({
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
                        },
                        **{
                           'must_not': [{'match': {'NOMEN_LONG.french': {'query': 'amicale du OR ass or association OR foyer OR sportive OR parents OR MAISON DES'}}
                                     }]
                        })
                    #,
#               'filter': [{'match': {'NOMEN_LONG.french': {'query': 'Lycee'}}
#                         }],                    
                  }
           }
    return body

def compute_threshold(metrics, t_p=0.95, t_r=0.3):
    ''' 
    Compute the optimal threshold and the associated metrics 
    
    INPUT:
        - metrics: list of individual metrics for one query (result of compute_metrics)
        - t_p: target_precision
        - t_r: target_recall
        
    OUTPUT:
        - thresh: optimal threshold (#TODO: explain more)
        - precision
        - recall
        - ratio
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
    
    thresh = sorted_metrics[idx]['_score_first']
    precision = rolling_precision[idx]
    recall = rolling_recall[idx]
    ratio = rolling_ratio[idx]
    
    if sum(is_first_vect) == 0:
        return 10**3, 0, 0, 0
    else:
        return thresh, precision, recall, ratio


def calc_agg_query_metrics(query_metrics):
    '''
    Find optimal threshold for each query and compute the aggregated metrics.
    
    INPUT:
        - query_metrics: dict of all results of previous queries
            
    OUTPUT:
        - agg_query_metrics: aggregated metrics for each query in query_metrics
    '''
    
    agg_query_metrics = dict()
    for key, metrics in query_metrics.items():
        thresh, precision, recall, ratio = compute_threshold(metrics)
        agg_query_metrics[key] = dict()
        agg_query_metrics[key]['thresh'] = thresh
        agg_query_metrics[key]['other'] = precision
        agg_query_metrics[key]['other'] = recall
        agg_query_metrics[key]['ratio'] = ratio
    return agg_query_metrics

    
    
def gen_label(full_responses, sorted_keys, row, num_results, num_rows_labelled):
    '''
    User labelling going through potential results (order given by sorted_keys) 
    looking for a match. This goes through all keys and all results (up to
    num_results) and asks the user to label the data. Labelling ends once the 
    user finds a match or there is no more potential matches.
    
    INPUT:
        - full_responses: result of "perform_queries" ; {query: response, ...}
        - sorted_keys: order in which to perform labelling
        - row: pandas.Series corresponding to the row being searched for
        - num_rows_labelled: For display only
        
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

def make_bulk(search_templates, num_results, chunk_size=100):
    '''
    Create a bulk generator with all search templates
    
    INPUT:
        - search_templates: iterator of form ((query_template, row), ...)
        - num_results: max num results per individual query
        - chunk_size: number of queries per bulk
    
    OUTPUT:
        - bulk_body: string containing queries formated for ES
        - queries: list of queries
    
    '''
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
            
    if bulk_body:
        yield bulk_body, queries

def perform_queries(all_query_templates, rows, num_results=3):
    '''
    Searches for the values in row with all the search templates in 
    all_query_templates. Retry on error.
    
    INPUT:
        - all_query_templates: iterator of queries to perform
        - rows: iterator of pandas.Series containing the rows to match
        - num_results: max number of results per individual query    
    '''
    i = 1
    full_responses = dict() 
    og_search_templates = list(enumerate(itertools.product(all_query_templates, rows)))
    search_templates = og_search_templates
    # search_template is [(id, (query, row)), ...]
    while search_templates:
        print('At search iteration', i)
        
        bulk_body_gen = make_bulk([x[1] for x in search_templates], num_results)
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
    '''
    Return concatenation of source and reference with the matches found
    
    INPUT:
        source: pandas.DataFrame containing all source items
        query_template: 
        threshold: minimum value of score for this query_template for a match
    '''
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
    


min_precision_tab = [(20, 0.7), (10, 0.5), (5, 0.3)] # (num_rows_labelled, min_prec)    
    

sorted_keys = all_query_templates

for num_rows_labelled, idx in enumerate(random.sample(list(source.index), num_samples)):
    row = source.loc[idx]
    print('Doing', row)  
        
    # Search elasticsearch for all results
    all_search_templates, full_responses = perform_queries(sorted_keys, [row], num_results_labelling)
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
            

        # Sort by ratio but label by precision ?
        sorted_keys = sorted(full_responses.keys(), key=lambda x: agg_query_metrics[x]['precision'], reverse=True)
        
        # Remove queries if precision is too low (thresh depends on number of labels)
        min_precision = 0
        for min_idx, min_precision in min_precision_tab:
            if num_rows_labelled >= min_idx:
                break
        sorted_keys = list(filter(lambda x: agg_query_metrics[x]['precision'] >= min_precision, sorted_keys))
    print('Number of keys left', len(sorted_keys))
    # Try to find the match somewhere

    # Make user label data until the match is found
    found, res = gen_label(full_responses, sorted_keys, row, num_results_labelling, num_rows_labelled)

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
    best_query_template = sorted(full_responses.keys(), key=lambda x: agg_query_metrics[x]['ratio'], reverse=True)[0]

new_source = match(source, best_query_template, 6)# agg_query_metrics[best_query_template]['thresh'])
new_source['has_match'] = new_source.__CONFIDENCE.notnull()

if test_num == 2:
    new_source['good'] = new_source.SIRET == (new_source.SIREN + new_source.NIC)
    print('Precision:', new_source.loc[new_source.has_match, 'good'].mean())
    print('Recall:', new_source.good.mean())

# Display results
new_source.sort_values('__CONFIDENCE', inplace=True)

for i, row in new_source.iloc[:100].iterrows(): 
    print('**** ({0}) / score: {1}\n'.format(i, row['__CONFIDENCE']))
    for match in match_cols:
        if isinstance(match['ref'], str):
            cols = [match['ref']]
        else:
            cols = match['ref']
        for col in cols:
            print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
            print('> {1}   -> [{0}]'.format(col, row[col]))  
        
if test_num == 2:
    for i, row in new_source[new_source.has_match & ~new_source.good].iloc[:100].iterrows(): 
        print('**** ({0}) / score: {1}\n'.format(i, row['__CONFIDENCE']))
        for match in match_cols + [{'source': 'SIRET', 'ref': ('SIREN', 'NIC')}]:
            if isinstance(match['ref'], str):
                cols = [match['ref']]
            else:
                cols = match['ref']
            for col in cols:
                print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
                print('> {1}   -> [{0}]'.format(col, row[col]))  
#            
#        for match in match_cols:
#            print(row[match['source']], '\n', row[match['ref']], '\n')

assert False

# Number of results
len(query_metrics[list(query_metrics.keys())[0]])

# Sorted metrics
for x in sorted_keys[:40]:
    print(x, '\n -> ', agg_query_metrics[x], '\n')

t_ps = [x/400. for x in range(4*80, 4*100)]
precs = []
recs = []
ratios = []

for t_p in t_ps:
    tab = pd.DataFrame([(key, *compute_threshold(metrics, t_p=t_p, t_r=0)) for key, metrics in query_metrics.items() if key in sorted_keys])
    tab.columns = ['key', 'thresh', 'precision', 'recall', 'ratio']
    tab.sort_values('ratio', ascending=False, inplace=True)
    precs.append(tab.precision.iloc[0])
    recs.append(tab.recall.iloc[0])
    ratios.append(tab.ratio.iloc[0])
    
import matplotlib.pyplot as plt
plt.plot(t_ps, precs)
plt.plot(t_ps, recs)
plt.plot(t_ps, ratios)
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
