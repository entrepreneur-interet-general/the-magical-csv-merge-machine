#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:39:55 2017

@author: m75380
"""
import itertools
import json
import random

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import unidecode

es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
 
def my_unidecode(string):
    '''unidecode or return empty string'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''

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

def _gen_body(query_template, row, must_not={}, must={}, num_results=3):
    '''
    Generate the string to pass to Elastic search for it to execute query
    
    INPUT:
        - query_template: ((source_col, ref_col, analyzer_suffix, boost), ...)
        - row: pandas.Series from the source object
        - must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
        - must: terms to filter by field (AND: will include ONLY IF ALL are in text)
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
                           'must_not': [{'match': {field: {'query': ' OR '.join(must_not)}}
                                     } for field, values in must_not.items()],
                           'filter': [{'match': {field: {'query': ' AND '.join(values)}}
                                     } for field, values in must.items()],
                        })               
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
        agg_query_metrics[key]['precision'] = precision
        agg_query_metrics[key]['recall'] = recall
        agg_query_metrics[key]['ratio'] = ratio
    return agg_query_metrics


def gen_all_query_templates(match_cols, columns_to_index, bool_levels, 
                            boost_levels, max_num_levels):
    ''' Generate query templates #TODO: more doc '''
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
    return all_query_templates    
    
#def gen_label(full_responses, sorted_keys, row, num_results, num_rows_labelled):
#    '''
#    User labelling going through potential results (order given by sorted_keys) 
#    looking for a match. This goes through all keys and all results (up to
#    num_results) and asks the user to label the data. Labelling ends once the 
#    user finds a match or there is no more potential matches.
#    
#    INPUT:
#        - full_responses: result of "perform_queries" ; {query: response, ...}
#        - sorted_keys: order in which to perform labelling
#        - row: pandas.Series corresponding to the row being searched for
#        - num_rows_labelled: For display only
#        
#    OUTPUT:
#        - found: If a match was found
#        - res: result of the match if it was found
#    '''
#    
#    ids_done = []
#    for key in sorted_keys:
#        print('\nkey: ', key)
#        results = full_responses[key]['hits']['hits']
#        for res in results[:num_results]:
#            if res['_id'] not in ids_done and ((res['_score']>=0.001)):
#                ids_done.append(res['_id'])
#                print('\n***** {0} / {1} / ({2})'.format(res['_id'], res['_score'], num_rows_labelled))
#                for match in match_cols:
#                    if isinstance(match['ref'], str):
#                        cols = [match['ref']]
#                    else:
#                        cols = match['ref']
#                    for col in cols:
#                        print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
#                        print('> {1}   -> [{0}]'.format(col, res['_source'][col]))
#                
#                if test_num == 2:
#                    print(row['SIRET'][:-5], row['SIRET'][-5:], '[source]')
#                    print(res['_source']['SIREN'], res['_source']['NIC'], '[ref]')
#                    print('LIBAPET', res['_source']['LIBAPET'])
#                    is_match = row['SIRET'] == res['_source']['SIREN'] + res['_source']['NIC']
#                else:
#                    is_match = input('Is match?\n > ') in ['1', 'y']
#                    
#                if is_match:
#                    return True, res
#                else:
#                    print('not first')
#    return False, None    

def make_bulk(table_name, search_templates, must, must_not, num_results, chunk_size=100):
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
        body = _gen_body(q_t, row, must, must_not, num_results)
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

def perform_queries(table_name, all_query_templates, rows, must, must_not, num_results=3):
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
        
        bulk_body_gen = make_bulk(table_name, [x[1] for x in search_templates], must, must_not, num_results)
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

def match(table_name, source, query_template, threshold):
    '''
    Return concatenation of source and reference with the matches found
    
    INPUT:
        source: pandas.DataFrame containing all source items
        query_template: 
        threshold: minimum value of score for this query_template for a match
    '''
    rows = (x[1] for x in source.iterrows())
    all_search_templates, full_responses = perform_queries(table_name, [query_template], rows, must, must_not, num_results=1)
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


class Labeller():
    max_num_samples = 100
    min_precision_tab = [(20, 0.7), (10, 0.5), (5, 0.3)]
    num_results_labelling = 3
    
    max_num_levels = 3 # Number of match clauses
    bool_levels = {'.integers': ['must', 'should']}
    boost_levels = [1]
    

    def __init__(self, source, ref_table_name, match_cols, columns_to_index, must={}, must_not={}):
        self.all_query_templates = gen_all_query_templates(match_cols, 
                                                           columns_to_index, 
                                                           self.bool_levels, 
                                                           self.boost_levels, 
                                                           self.max_num_levels)
        self.sorted_keys = self.all_query_templates
        self.source = source
        self.ref_table_name = ref_table_name
        
        self.match_cols = match_cols
        self.num_rows_labelled = 0
        self.query_metrics = dict()
        
        self.must = must
        self.must_not = must_not
        
        for q_t in self.all_query_templates:
            self.query_metrics[q_t] = []
            
        self.row_idxs = (idx for idx in random.sample(list(source.index), self.max_num_samples))
        self.next_row = True
    
    def _min_precision(self):
        min_precision = 0
        for min_idx, min_precision in self.min_precision_tab:
            if self.num_rows_labelled >= min_idx:
                break    
        return min_precision

    def _user_input(self, res, row, test_num=0):
        print('\n***** {0} / {1} / ({2})'.format(res['_id'], res['_score'], self.num_rows_labelled))
        for match in self.match_cols:
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
            if row['SIRET'] == res['_source']['SIREN'] + res['_source']['NIC']:
                return 'y'
            else:
                return 'n'
        else:
            return input('Is match?\n > ')
        
    def new_label_row(self, full_responses, sorted_keys, num_results):
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
                    yield res

    def new_label(self):
        # If looking for a new row, initiate the generator
        if self.next_row: # If previous was found: try new row
            self.next_row = False
            try:
                self.idx = next(self.row_idxs)
            except StopIteration:
                return False
            self.row = self.source.loc[self.idx]
            
            print('in new_label / in self.next_row / len sorted_keys: {0}'.format(len(self.sorted_keys)))
            self.all_search_templates, self.full_responses = perform_queries(self.ref_table_name, self.sorted_keys, [self.row], self.must, self.must_not, self.num_results_labelling)
            self.full_responses = {self.all_search_templates[idx][1][0]: values for idx, values in self.full_responses.items()}

            # Sort keys by score or most promising
            if self.num_rows_labelled <= 2:
                sorted_keys_1 = random.sample(list(self.full_responses.keys()), len(self.full_responses.keys()))
                sorted_keys_2 = sorted(self.full_responses.keys(), key=lambda x: self.full_responses[x]['hits'].get('max_score') or 0, reverse=True)
                self.sorted_keys = list(itertools.chain(*zip(sorted_keys_1, sorted_keys_2)))
            else:
                # Sort by ratio but label by precision ?
                self.sorted_keys = sorted(self.full_responses.keys(), key=lambda x: self.agg_query_metrics[x]['precision'], reverse=True)
                
                # Remove queries if precision is too low (thresh depends on number of labels)
                self.sorted_keys = list(filter(lambda x: self.agg_query_metrics[x]['precision'] \
                                          >= self._min_precision(), self.sorted_keys))            
                
            self.label_row_gen = self.new_label_row(self.full_responses, self.sorted_keys, self.num_results_labelling)
        
        try:
            return next(self.label_row_gen)
        except StopIteration:
            self.next_row = True
            return self.new_label()
    
    def parse_valid_answer(self, user_input):
        is_match = user_input in ['y', '1']  
        return is_match

    def update(self, is_match, res_id):
        print('in update')
        if is_match:
            print('in update / in is_match')
            # Update individual and aggregated metrics
            for key, response in self.full_responses.items():
                self.query_metrics[key].append(compute_metrics(response['hits']['hits'], res_id))
            self.agg_query_metrics = calc_agg_query_metrics(self.query_metrics)
            
            self.num_rows_labelled += 1
            self.next_row = True


    def answer_is_valid(self, user_input):
        '''Check if the user input is valid'''
        valid_responses = {'y', 'n', 'u', 'f', '0', '1'}
        #        if self.examples_buffer:
        #            valid_responses = {'y', 'n', 'u', 'f', 'p'}
        #        else: 
        #            valid_responses = {'y', 'n', 'u', 'f'}
        return user_input in valid_responses


    def to_emit(self, message):
        '''Creates a dict to be sent to the template'''
        dict_to_emit = dict()
        dict_to_emit['formated_record_pair'] = self._format_record_pair()
        dict_to_emit['formated_example'] = self._format_fields() # TODO: remove this
        dict_to_emit['n_match'] = str(self.n_match)
        dict_to_emit['n_distinct'] = str(self.n_distinct)
        dict_to_emit['has_previous'] = len(self.examples_buffer) >= 1
        if message:
            dict_to_emit['_message'] = message
        return dict_to_emit
            

    def cleanup_training(self):
        self.deduper.cleanupTraining()

    def write_training(self, file_path):
        with open(file_path, 'w') as f:
            self.deduper.writeTraining(f)

