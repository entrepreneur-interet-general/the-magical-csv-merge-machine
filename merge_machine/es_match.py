#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:39:55 2017

@author: m75380

Add extend sorted keys

Add option to label full file (no inference on unlabelled)

multiple query

security 

big problem with add_selected_columns and cleaning of previous file

Discrepancy btw labelling (worse) and matching (better)

Problem with precision when no match found during labelling

--> add (source_id, None) to pairs when none are found

"""
from collections import defaultdict
import copy
import itertools
import json
import random

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import unidecode

from my_json_encoder import MyEncoder

es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
 
def my_unidecode(string):
    '''unidecode or return empty string'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''

def analyze_hits(hits, target_ref_id):
    '''
    Computes a summary for based on the : res['hits']['hits']
    
    INPUT:
        - hits: res['hits']['hits']
        - ref_id: The real target Elasticsearch ID
        
    OUTPUT:
        - summary: dict with various information    
    '''
    
    # General query summary
    num_hits = len(hits)

    if num_hits:
        _score_first = hits[0]['_score']
    else:
        _score_first = 0
    
    # Match summary
    try:
        i = next((i for i, hit in enumerate(hits) if hit['_id']==target_ref_id))
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
        
    summary = dict()
    summary['_score_first'] = _score_first
    summary['num_hits'] = num_hits
    summary['has_match'] = has_match
    summary['is_first'] = is_first
    summary['es_id_score'] = es_id_score
    summary['pos_score'] = pos_score
    summary['target_id'] = target_ref_id
    
    return summary

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
    else:
        raise ValueError('s_q_t_2 should be str or tuple (not list)')
    yield '' # No suffix for standard analyzer
    for analyzer in analyzers:
        yield '.' + analyzer

def _remove_words(string, words):
    # TODO: fix this
    string = my_unidecode(string).lower()
    for word in words:
        string = string.replace(word, '')
    return string

def _reformat_s_q_t(s_q_t):
    '''Makes sure s_q_t[1] is a list'''
    if isinstance(s_q_t[1], str):
        old_len = len(s_q_t)
        s_q_t = (s_q_t[0], [s_q_t[1]], s_q_t[2], s_q_t[3], s_q_t[4])
        assert len(s_q_t) == old_len
    elif isinstance(s_q_t[1], list) or isinstance(s_q_t[1], tuple):
        s_q_t = (s_q_t[0], list(s_q_t[1]), s_q_t[2], s_q_t[3], s_q_t[4])
    else:
        raise ValueError('Single query template element 1 should be str or list')
    return s_q_t

def _gen_body(query_template, row, must={}, must_not={}, num_results=3):
    '''
    Generate the string to pass to Elastic search for it to execute query
    
    INPUT:
        - query_template: ((bool_lvl, source_col, ref_col, analyzer_suffix, boost), ...)
        - row: pandas.Series from the source object
        - must: terms to filter by field (AND: will include ONLY IF ALL are in text)
        - must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
        - num_results: Max number of results for the query
    
    OUTPUT:
        - body: the query as string
    
    NB: s_q_t: single_query_template
        source_val = row[s_q_t[1]]
        key = s_q_t[2] + s_q_t[3]
        boost = s_q_t[4]
    '''
    DEFAULT_MUST_FIELD = '.french'
    
    query_template = [_reformat_s_q_t(s_q_t) for s_q_t in query_template]
    
    body = {
          'size': num_results,
          'query': {
            'bool': dict({
               must_or_should: [
                          {'match': {
                                  s_q_t[2] + s_q_t[3]: {'query': _remove_words(row[s_q_t[1]].str.cat(sep=' '), must.get(s_q_t[2], [])),
                                                        'boost': s_q_t[4]}}
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and isinstance(s_q_t[2], str)
                        ] \
    
                        + [
                          {'multi_match': {
                                  'fields': [col + s_q_t[3] for col in s_q_t[2]], 
                                  'query': _remove_words(row[s_q_t[1]].str.cat(sep=' '), []),
                                  'boost': s_q_t[4]
                                  }
                          } \
                          for s_q_t in query_template if (s_q_t[0] == must_or_should) \
                                      and (isinstance(s_q_t[2], tuple) or isinstance(s_q_t[2], list))
                        ] \
                for must_or_should in ['must', 'should']
                },
    
                    **{
                       'must_not': [{'match': {field + DEFAULT_MUST_FIELD: {'query': ' OR '.join(values)}}
                                 } for field, values in must_not.items()],
                       'filter': [{'match': {field + DEFAULT_MUST_FIELD: {'query': ' AND '.join(values)}} # TODO: french?
                                 } for field, values in must.items()],
                    })               
                  }
           }
    return body

def compute_threshold(summaries, t_p=0.95, t_r=0.3):
    ''' 
    Compute the optimal threshold and the associated metrics 
    
    INPUT:
        - summaries: list of individual summaries for one query (result of analyze_hits)
        - t_p: target_precision
        - t_r: target_recall
        
    OUTPUT:
        - thresh: optimal threshold (#TODO: explain more)
        - precision
        - recall
        - ratio
    '''
    num_summaries = len(summaries)

    # Scores deal with empty hits
    sorted_summaries = sorted(summaries, key=lambda x: x['_score_first'], reverse=True)
    
    # score_vect = np.array([x['_score_first'] for x in sorted_summaries])
    is_first_vect = np.array([bool(x['is_first']) for x in sorted_summaries])
    # rolling_score_mean = score_vect.cumsum() / (np.arange(num_summaries) + 1)
    rolling_precision = is_first_vect.cumsum() / (np.arange(num_summaries) + 1)
    rolling_recall = is_first_vect.cumsum() / num_summaries
    
    # WHY is recall same as precision ?
    
    # Compute ratio
    _f_precision = lambda x: max(x - t_p, 0) + min(t_p*(x/t_p)**3, t_p)
    _f_recall = lambda x: max(x - t_r, 0) + min(t_p*(x/t_r)**3, t_r)
    a = np.fromiter((_f_precision(xi) for xi in rolling_precision), rolling_precision.dtype)
    b = np.fromiter((_f_recall(xi) for xi in rolling_recall), rolling_recall.dtype)
    rolling_ratio = a*b

    # Find best index for threshold
    idx = max(num_summaries - rolling_ratio[::-1].argmax() - 1, min(6, num_summaries-1))
    
    thresh = sorted_summaries[idx]['_score_first']
    precision = rolling_precision[idx]
    recall = rolling_recall[idx]
    ratio = rolling_ratio[idx]
    
    if sum(is_first_vect) == 0:
        return 10**3, 0, 0, 0
    else:
        return thresh, precision, recall, ratio


def calc_query_metrics(query_summaries, t_p=0.95, t_r=0.3):
    '''
    Find optimal threshold for each query and compute the aggregated metrics.
    
    INPUT:
        - query_summaries: dict of all results of previous queries
            
    OUTPUT:
        - query_metrics: aggregated metrics for each query in query_summaries
    '''
    
    query_metrics = dict()
    for key, summaries in query_summaries.items():
        thresh, precision, recall, ratio = compute_threshold([x[1] for x in summaries], t_p=0.95, t_r=0.3) #TODO: change name in compute_threshold
        query_metrics[key] = dict()
        query_metrics[key]['thresh'] = thresh
        query_metrics[key]['precision'] = precision
        query_metrics[key]['recall'] = recall
        query_metrics[key]['ratio'] = ratio

    return query_metrics

def DETUPLIFY_TODO_DELETE(arg):
    if isinstance(arg, tuple) and (len(arg) == 1):
        return arg[0]
    return arg

def _gen_all_query_templates(match_cols, columns_to_index, bool_levels, 
                            boost_levels, max_num_levels):
    ''' Generate query templates #TODO: more doc '''
    single_queries = list(((bool_lvl, DETUPLIFY_TODO_DELETE(x['source']), DETUPLIFY_TODO_DELETE(x['ref']), suffix, boost) \
                                       for x in match_cols \
                                       for suffix in _gen_suffix(columns_to_index, x['ref']) \
                                       for bool_lvl in bool_levels.get(suffix, ['must']) \
                                       for boost in boost_levels))
    all_query_templates = list(itertools.chain(*[list(itertools.combinations(single_queries, x)) \
                                        for x in range(2, max_num_levels+1)][::-1]))
    # Queries must contain at least two distinct columns (unless one match is mentionned)
    if len(match_cols) > 1:
        all_query_templates = list(filter(lambda query: len(set((x[1], x[2]) for x in query)) >= 2, \
                                    all_query_templates))
    return all_query_templates    
    
def _expand_by_boost(all_query_templates):
    '''
    For each query template, and for each single query template within, generate
    a new compound query template by doubling the boost level of a single query
    template (and normalizing)
    '''
    new_query_templates = copy.deepcopy(all_query_templates)
    
    for compound_query in all_query_templates:
        if len(compound_query) >= 2:
            old_boost = [x[4] for x in compound_query]
            for i in range(len(compound_query)):
                new_boost = copy.deepcopy(old_boost)
                new_boost[i] *= 2
                # TODO: do not expand if boost is already large 
                new_boost = [x/sum(new_boost) * len(new_boost) for x in new_boost]
                new_compound_query = tuple((x[0], x[1], x[2], x[3], b) for x, b in zip(compound_query, new_boost))
                if new_compound_query not in new_query_templates:
                    new_query_templates.append(new_compound_query)
    return new_query_templates
    
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

def _gen_bulk(table_name, search_templates, must, must_not, num_results, chunk_size=100):
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
        #        if i == 0:
        #            print(body)
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
    search_templates = list(og_search_templates)        
    # search_template is [(id, (query, row)), ...]
    while search_templates:
        print('At search iteration', i)
        
        bulk_body_gen = _gen_bulk(table_name, [x[1] for x in search_templates], 
                                  must, must_not, num_results)
        responses = []
        for bulk_body, _ in bulk_body_gen:
            responses.extend(es.msearch(bulk_body)['responses']) #, index=table_name)
            
        # TODO: add error on query template with no must or should
        
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
            import pdb; pdb.set_trace()
            raise Exception('Problem with elasticsearch: could not perform all queries in 10 trials')
            
    return og_search_templates, full_responses

#def exact_es_linker(source, params):
#    table_name = params['index_name']
#    certain_col_matches = params['certain_col_matches']
#    exact_pairs = params.get('exact_pairs', [])
#    
#    exact_source_indexes = [x[0] for x in exact_pairs]
#    source_indexes = (x[0] for x in source.iterrows() if x [0] not in exact_source_indexes)    
#    
#    query_template = ('must', certain_col_matches['source'], certain_col_matches['ref'], '', 1)
#
#    rows = (x[1] for x in source.iterrows() if x[0] not in exact_source_indexes)
#    all_search_templates, full_responses = perform_queries(table_name, [query_template], rows, [], [], num_results=1)    
#    full_responses = [full_responses[i] for i in range(len(full_responses))] # Don't use items to preserve order
#    
#    matches_in_ref = pd.DataFrame([f_r['hits']['hits'][0]['_source'] \
#                               if bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold) \
#                               else {} \
#                               for f_r in full_responses], index=source_indexes)
#                    
#    confidence = pd.Series([f_r['hits']['hits'][0]['_score'] \
#                            if bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold) \
#                            else np.nan \
#                            for f_r in full_responses], index=matches_in_ref.index)
#    matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
#    matches_in_ref['__CONFIDENCE'] = 998    
    
    
    
def es_linker(source, params):
    '''
    Return concatenation of source and reference with the matches found
    
    INPUT:
        source: pandas.DataFrame containing all source items
        params:    
            index_name: name of the Elasticsearch index to fetch from
            query_template: 
            threshold: minimum value of score for this query_template for a match
            must: terms to filter by field (AND: will include ONLY IF ALL are in text)
            must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
            exact_pairs: list of (source_id, ES_ref_id) which are certain matches
    '''
    
    table_name = params['index_name']
    query_template = params['query_template']
    must = params.get('must', {})
    must_not = params.get('must_not', {})
    threshold = params['thresh']
    exact_pairs = params.get('exact_pairs', [])
    
    exact_source_indexes = [x[0] for x in exact_pairs if x[1] is not None]
    exact_ref_indexes = [x[1] for x in exact_pairs if x[1] is not None]
    source_indexes = (x[0] for x in source.iterrows() if x [0] not in exact_source_indexes)    
    
    def _is_match(f_r, threshold):
        bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold)
    
    # Perform matching on non-exact pairs (not labelled)
    if source_indexes:
        rows = (x[1] for x in source.iterrows() if x[0] not in exact_source_indexes)
        all_search_templates, full_responses = perform_queries(table_name, [query_template], rows, must, must_not, num_results=2)
        full_responses = [full_responses[i] for i in range(len(full_responses))] # Don't use items to preserve order
    
        matches_in_ref = pd.DataFrame([f_r['hits']['hits'][0]['_source'] \
                                   if _is_match(f_r, threshold) \
                                   else {} \
                                   for f_r in full_responses], index=source_indexes)
                        
        ref_id = pd.Series([f_r['hits']['hits'][0]['_id'] \
                                if _is_match(f_r, threshold) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)
    
        confidence = pd.Series([f_r['hits']['hits'][0]['_score'] \
                                if _is_match(f_r, threshold) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)

        confidence_gap = pd.Series([f_r['hits']['hits'][0]['_score'] - f_r['hits']['hits'][1]['_score']
                                if _is_match(f_r, threshold) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)

        matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
        matches_in_ref['__ID_REF'] = ref_id
        matches_in_ref['__CONFIDENCE'] = confidence    
        matches_in_ref['__GAP'] = confidence_gap
        matches_in_ref['__GAP_RATIO'] = confidence_gap / confidence
    else:
        matches_in_ref = pd.DataFrame()
        
    # Perform matching exact (labelled) pairs
    if exact_ref_indexes:
        full_responses = [es.get(table_name, ref_idx) for ref_idx in exact_ref_indexes]
        exact_matches_in_ref = pd.DataFrame([f_r['_source'] for f_r in full_responses], 
                                            index=exact_source_indexes)
        exact_matches_in_ref.columns = [x + '__REF' for x in exact_matches_in_ref.columns]
        exact_matches_in_ref['__ID_REF'] = exact_ref_indexes
        exact_matches_in_ref['__CONFIDENCE'] = 999
    else:
        exact_matches_in_ref = pd.DataFrame()
    
    #
    assert len(exact_matches_in_ref) + len(matches_in_ref) == len(source)
    new_source = pd.concat([source, pd.concat([matches_in_ref, exact_matches_in_ref])], 1)        
    
    modified = new_source.copy() # TODO: is this good?
    modified.loc[:, :] = True
    return new_source, modified



class Labeller():
    max_num_samples = 100
    min_precision_tab = [(20, 0.7), (10, 0.5), (5, 0.3)]
    max_num_keys_tab = [(20, 10), (10, 50), (7, 200), (5, 500), (0, 1000)]
    num_results_labelling = 3
    
    max_num_levels = 3 # Number of match clauses
    bool_levels = {'.integers': ['must', 'should'], 
                   '.city': ['must', 'should']}
    boost_levels = [1]
    
    t_p = 0.95
    t_r = 0.3
    
    def __init__(self, source, ref_table_name, match_cols, columns_to_index, 
                 certain_column_matches=None, must={}, must_not={}):
        all_query_templates = _gen_all_query_templates(match_cols, 
                                                           columns_to_index, 
                                                           self.bool_levels, 
                                                           self.boost_levels, 
                                                           self.max_num_levels)
        

        self.source = source
        self.ref_table_name = ref_table_name
        
        self.match_cols = match_cols
        self.certain_column_matches = certain_column_matches
        self.num_rows_proposed_source = 0
        self.num_rows_proposed_ref = defaultdict(int)
        self.num_rows_labelled = 0
        self.query_summaries = dict()
        self.query_metrics = dict()

        d_q = self._default_query()
        all_query_templates = [d_q] + [x for x in all_query_templates if x != d_q]
        
        for q_t in all_query_templates:
            self.query_summaries[q_t] = []
        
        # TODO: Must currently not implemented
        self.must = must
        self.must_not = must_not
            
        self.row_idxs = list(idx for idx in random.sample(list(source.index), self.max_num_samples))
        self.pairs = [] # List of (source_id, ref_es_id)
        self.pairs_not_match = defaultdict(list)
        self.next_row = True
        self.finished = False

        self.sorted_keys = all_query_templates
        
        if certain_column_matches is not None:
            self.auto_label()

    
    def _min_precision(self):
        '''
        Return the minimum precision to keep a query template, according to 
        the number of rows currently labelled
        '''
        min_precision = 0
        for min_idx, min_precision in self.min_precision_tab:
            if self.num_rows_labelled >= min_idx:
                break    
        return min_precision

    def _max_num_keys(self):
        '''
        Max number of labels based on the number of rows currently labelled
        '''
        max_num_keys = self.max_num_keys_tab[-1][1]
        for min_idx, max_num_keys in self.max_num_keys_tab[:-1]:
            if self.num_rows_labelled >= min_idx:
                break    
        return max_num_keys
        

    def _print_pair(self, source_item, ref_item, test_num=0):
        print('\n***** ref_id: {0} / score: {1} / num_rows_labelled: {2}'.format(ref_item['_id'], ref_item['_score'], self.num_rows_labelled))
        print('self.first_propoal_for_source_idx:', self.first_propoal_for_source_idx)
        print('self.num_rows_labelled:', self.num_rows_labelled)
        print('self.num_rows_proposed_source:', self.num_rows_proposed_source)
        print('self.num_rows_proposed_ref:', self.num_rows_proposed_ref)
        match_cols = copy.deepcopy(self.match_cols)
        for match in match_cols:
            if isinstance(match['ref'], str):
                cols = [match['ref']]
            else:
                cols = match['ref']
            for col in cols:
                if isinstance(match['source'], str):
                    match['source'] = [match['source']]
                string = ' '.join([source_item['_source'][col_source] for col_source in match['source']])
                print('\n{1}   -> [{0}][source]'.format(match['source'], string))
                print('> {1}   -> [{0}]'.format(col, ref_item['_source'][col]))
        
    def console_input(self, source_item, ref_item, test_num=0):
        '''Console input displaying the source_item, ref_item pair'''
        self._print_pair(source_item, ref_item, test_num)
        if test_num == 2:
            print(source_item['_source']['SIRET'][:-5], source_item['_source']['SIRET'][-5:], '[source]')
            print(ref_item['_source']['SIREN'], ref_item['_source']['NIC'], '[ref]')
            print('LIBAPET', ref_item['_source']['LIBAPET'])
            if source_item['_source']['SIRET'] == ref_item['_source']['SIREN'] + ref_item['_source']['NIC']:
                return 'y'
            else:
                return 'n'
        else:
            return input('Is match?\n > ')


    def _default_query(self):
        '''
        The default query to be performed in the first few rounds of labelling
        '''        
        if len(self.match_cols) == 1:
            m = self.match_cols[0]
            return (('must', m['source'], m['ref'], '.french', 1), ('should', m['source'], m['ref'], '.integers', 1))
        else:
            return tuple(('must', m['source'], m['ref'], '.french', 1) for m in self.match_cols)
        

    def _sort_keys(self):
        '''
        Update sorted_keys, that determin the order in which samples are shown
        to the user        
        '''
        
        # Sort keys by score or most promising
        if self.num_rows_labelled <= 2:
            # Alternate between random and largest score
            sorted_keys_1 = random.sample(list(self.full_responses.keys()), len(self.full_responses.keys()))
            sorted_keys_2 = sorted(self.full_responses.keys(), key=lambda x: \
                                   self.full_responses[x]['hits'].get('max_score') or 0, reverse=True)
            
            
            d_q = self._default_query()
            self.sorted_keys =  [d_q] \
                        + [x for x in list(itertools.chain(*zip(sorted_keys_1, sorted_keys_2))) if x != d_q]
            # TODO: redundency with first
        else:
            # Sort by ratio but label by precision ?
            self.sorted_keys = sorted(self.full_responses.keys(), key=lambda x: self.query_metrics[x]['precision'], reverse=True)
            
            # Remove queries if precision is too low (thresh depends on number of labels)
            self.sorted_keys = list(filter(lambda x: self.query_metrics[x]['precision'] \
                                      >= self._min_precision(), self.sorted_keys))
            
            self.sorted_keys  = self.sorted_keys[:self._max_num_keys()]

    def previous(self):
        '''Return to pseudo-previous state.'''
        print('self.next_row:', self.next_row)
        if self.first_propoal_for_source_idx:
            self._previous_row()
        else:
            self._restart_row()        
    
    def _restart_row(self):
        '''Re-initiates labelling for row self.idx'''       
        if not hasattr(self, 'idx'):
            raise RuntimeError('No row to restart')
        if self.next_row:
            raise RuntimeError('Already at start of row')
        
        self.num_rows_proposed_ref[self.idx] = 0 
          
        
        # after update
        if self.pairs and self.pairs[-1][0] == self.idx:
            self.num_rows_labelled -= 1
            self.pairs.pop()
            
            for val in self.query_summaries.values():
                if val and (val[-1][0] == self.idx):
                    val.pop()
                    
        self.num_rows_proposed_source -= 1     
        self.next_row = True
        
        self.row_idxs.append(self.idx)
        
        # self.query_summaries = dict()
        
            
    def _previous_row(self):
        '''Todo this deals with previous '''
        if not self.pairs:
            raise RuntimeError('No previous labels')
        
        self._restart_row()
        
        previous_idx = self.pairs.pop()[0]
        self.row_idxs.append(previous_idx)
        self.num_rows_labelled -= 1

        self.num_rows_proposed_ref[previous_idx] = 0
        self.num_rows_proposed_source -= 1
        
        # TODO: remove try, except
        try:
            self._update_query_metrics()       
        except:
            print('Could not update query_metrics')
            pass        
        print('self.next_row:', self.next_row)

        
    def _new_label_row(self, full_responses, sorted_keys, num_results):
        '''
        User labelling going through potential results (order given by sorted_keys) 
        looking for a match. This goes through all keys and all results (up to
        num_results) and asks the user to label the data. Labelling ends once the 
        user finds a match for the current row or there is no more potential matches.
        
        INPUT:
            - full_responses: result of "perform_queries" ; {query: response, ...}
            - sorted_keys: order in which to perform labelling
            - num_results: how many results to display per search template (1
                        will display only most probable result for each template)
            
        OUTPUT:
            - res: result of potential match in reference to label
        '''
        
        ids_done = []
        num_keys = len(sorted_keys)
        for i, key in enumerate(sorted_keys):
            try:
                print('precision: ', self.query_metrics[key]['precision'])
                print('recall: ', self.query_metrics[key]['recall'])
            except:
                print('no precision/recall to display...')
                
            results = full_responses[key]['hits']['hits']
            
            if len(results):
                print('\nkey ({0}/{1}): {2}'.format(i, num_keys, key))
                print('Num hits for this key: ', len(results)) 
            else:
                print('\nkey ({0}/{1}) has no results...'.format(i, num_keys))
            for res in results[:num_results]:
                min_score = self.query_metrics.get(key, {}).get('thresh', 0)/1.5
                if res['_id'] not in ids_done \
                        and ((res['_score']>=min_score)) \
                        and res['_id'] not in self.pairs_not_match[self.idx]: #  TODO: not neat
                    ids_done.append(res['_id'])
                    yield res
                else:
                    print('>>> res not analyzed because:')
                    if res['_id'] in ids_done: 
                        print('> res already done for this row')
                    if res['_score'] < min_score:
                        print('> score too low ({0} < {1})'.format(res['_score'], min_score))
                    if res['_id'] in self.pairs_not_match[self.idx]: 
                        print('> already a NOT match')

    
    def _new_label(self):
        '''Return the next potential match in reference to label to label'''
        # If looking for a new row from source, initiate the generator
        if not self.sorted_keys:
            raise ValueError('No keys in self.sorted_keys')
            
        self.first_propoal_for_source_idx = False
        # If on a new row, create the generator for the entire row
        if self.next_row: # If previous was found: try new row
            if self.row_idxs:
                self.idx = self.row_idxs.pop()
                
                # Check if row was already done # TODO: will be problem with count
                if self.idx in (x[0] for x in self.pairs):
                    return self._new_label() # TODP! chechk this                     
                
                self.first_propoal_for_source_idx = True
            else:
                self.finished = True
                return None            
            
            
            self.num_rows_proposed_source += 1
            self.next_row = False
            
            row = self.source.loc[self.idx]
            
            print('\n' + '*'*40+'\n', 'in new_label / in self.next_row / len sorted_keys: {0} / row_idx: {1}'.format(len(self.sorted_keys), self.idx))
            all_search_templates, tmp_full_responses = perform_queries(self.ref_table_name, self.sorted_keys, [row], self.must, self.must_not, self.num_results_labelling)
            self.full_responses = {all_search_templates[i][1][0]: values for i, values in tmp_full_responses.items()}
            print('LEN OF FULL_RESPONSES:', len(self.full_responses))
            # import pdb; pdb.set_trace()
            self._sort_keys()
                            
            self.label_row_gen = self._new_label_row(self.full_responses, 
                                                    self.sorted_keys, 
                                                    self.num_results_labelling)
        
        # Return next option for row or try next row
        try:
            self.num_rows_proposed_ref[self.idx] += 1
            return next(self.label_row_gen)
        except StopIteration:
            self.next_row = True
            return self._new_label()
        
    def new_label(self):
        '''Returns a pair to label'''
        ref_item = self._new_label()
        
        if ref_item is None:
            return None, None
        
        source_item = {'_id': self.idx, 
                       '_source': self.source.loc[self.idx].to_dict()}
        
        self.source_item = source_item
        self.ref_item = ref_item
        return source_item, ref_item
    
    def _auto_label_pair(self, source_item, ref_item):
        '''
        Try to automatically generate the label for the pair source_item, ref_item
        '''
        # TODO: check format of exact match cols
        source_cols = self.certain_column_matches['source']
        ref_cols = self.certain_column_matches['ref']
        
        # If no values are None, check if concatenation is an exact match
        if all(source_item['_source'][col] is not None for col in source_cols) \
                and all(ref_item['_source'][col] is not None for col in ref_cols):
                    
            is_match = ''.join(source_item['_source'][col] for col in source_cols) \
                        == ''.join(ref_item['_source'][col] for col in ref_cols)
                        
            if is_match:
                return 'y'
            else:
                return 'n'
        else:
            return None
        
    def auto_label(self):
        for i in range(len(self.source)):
            (source_item, ref_item) = self.new_label()
            if ref_item is None:
                break
            
            label = self._auto_label_pair(source_item, ref_item)
            
            self._print_pair(source_item, ref_item, 0)
            print('AUTO LABEL: {0}'.format(label))
            
            if label is not None:
                self.update(label, ref_item['_id'])    
        
        self.row_idxs = list(idx for idx in random.sample(list(self.source.index), 
                                                          self.max_num_samples))
        self.next_row = True
        
    def parse_valid_answer(self, user_input):
        if self.ref_item is not None:
            ref_id = self.ref_item['_id']
        else:
            ref_id = None
        return self.update(user_input, ref_id)

    def update(self, user_input, ref_id):
        '''
        Update query summaries and query_metrics and other variables based 
        on the user given label and the elasticsearch id of the reference item being labelled
        
        INPUT:
            - user_input: 
                + "y" or "1" : res_id is a match with self.idx
                + "n" or "0" : res_id is not a match with self.idx
                + "u" : uncertain #TODO: is this no ?
                + "p" : back to previous state
            - ref_id: Elasticsearch Id of the reference element being labelled
        '''
        MIN_NUM_KEYS = 9 # Number under which to expand
        EXPAND_FREQ = 9
        
        use_previous = user_input == 'p'
        
        if use_previous:
            self.previous()
        else:
            is_match = user_input in ['y', '1']
            is_not_match = user_input in ['n', '0']
            print('in update')
            if is_match:
                
                if (self.pairs) and (self.pairs[-1][0] == self.idx):
                    self.pairs.pop()
                self.pairs.append((self.idx, ref_id))
    
                self._update_query_summaries(ref_id)
                self.num_rows_labelled += 1
                self.next_row = True
            if is_not_match:
                self.pairs_not_match[self.idx].append(ref_id)
                
                if (not self.pairs) or (self.pairs[-1][0] != self.idx):
                    self.pairs.append((self.idx, None))
                
#            if False:
            # TODO: num_rows_labelled is not good since it can not change on new label
            try:
                self.last_expanded
            except:
                self.last_expanded = None
            if ((len(self.sorted_keys) < MIN_NUM_KEYS) \
                or((self.num_rows_labelled+1) % EXPAND_FREQ==0)) \
                and (self.last_expanded != self.idx):
                self.sorted_keys = _expand_by_boost(self.sorted_keys)
                self.re_score_history()    
                self.last_expanded = self.idx
    
        
    def _update_query_metrics(self):
        self.query_metrics = calc_query_metrics(self.query_summaries, t_p=self.t_p, t_r=self.t_r)

    def _update_query_summaries(self, matching_ref_id):
        '''
        Assuming res_id is the Elasticsearch id of the matching refential,
        update the summaries
        '''

        print('in update / in is_match')
        # Update individual and aggregated summaries
        for key, response in self.full_responses.items():
            self.query_summaries[key].append([self.idx, analyze_hits(response['hits']['hits'], matching_ref_id)])
        self._update_query_metrics()
        
    def re_score_history(self):
        '''Use this if updated must or must_not'''
        print('WARNING: Re-Scoring history')

        # Re-initiate query_summaries
        self.query_summaries = dict()
        for q_t in self.sorted_keys:
            self.query_summaries[q_t] = []
        
        # TODO: temporary sol: put all in bulk
        for pair in self.pairs:
            all_search_templates, self.full_responses = perform_queries(
                                                          self.ref_table_name, 
                                                          self.sorted_keys, 
                                                          [self.source.loc[pair[0]]], 
                                                          self.must, self.must_not, 
                                                          self.num_results_labelling)
            self.full_responses = {all_search_templates[idx][1][0]: values for idx, values in self.full_responses.items()}
            self._update_query_summaries(pair[1])

        self._sort_keys()

        if not self.sorted_keys:
            raise ValueError('No keys in self.sorted_keys')
        
    def answer_is_valid(self, user_input):
        '''Check if the user input is valid'''
        valid_responses = {'y', 'n', 'u', '0', '1', 'p'}
        return user_input in valid_responses

    def export_best_params(self):
        '''Returns a dictionnary with the best parameters (input for es_linker)'''
        params = dict()
        params['index_name'] = self.ref_table_name
        params['query_template'] = self._best_query_template()
        params['must'] = self.must
        params['must_not'] = self.must_not
        params['thresh'] = 0 # self.threshold #TODO: fix this
        params['exact_pairs'] = self.pairs
        return params
    
    def write_training(self, file_path):        
        params = self.export_best_params()
        encoder = MyEncoder()
        with open(file_path, 'w') as w:
            w.write(encoder.encode(params))
    
    def update_musts(self, must, must_not):
        if (not isinstance(must, dict)) or (not isinstance(must_not, dict)):
            raise ValueError('Variables "must" and "must_not" should be dicts' \
                'with keys being column names and values a list of strings')
        self.must = must
        self.must_not = must_not
        self.re_score_history()
        
    def _best_query_template(self):
        """Return query template with the best score (ratio)"""
        if self.query_metrics:
            return sorted(self.query_metrics.keys(), key=lambda x: \
                          self.query_metrics[x]['ratio'], reverse=True)[0]
        else:
            return None
 
    def to_emit(self, message):
        '''Creates a dict to be sent to the template #TODO: fix this'''
        dict_to_emit = dict()
        # Info on pair
        dict_to_emit['source_item'] = self.source_item
        dict_to_emit['ref_item'] = self.ref_item
        
        # Info on past labelling
        dict_to_emit['num_proposed_source'] = str(self.num_rows_proposed_source)
        dict_to_emit['num_proposed_ref'] = str(sum(self.num_rows_proposed_ref.values()))
        dict_to_emit['num_labelled'] = str(self.num_rows_labelled)
        dict_to_emit['t_p'] = self.t_p
        dict_to_emit['t_r'] = self.t_r
        
        # Info on current performence
        b_q_t = self._best_query_template()
        if b_q_t is not None:
            dict_to_emit['estimated_precision'] = str(self.query_metrics.get(b_q_t, {}).get('precision'))
            dict_to_emit['estimated_recall'] = str(self.query_metrics.get(b_q_t, {}).get('recall'))
            dict_to_emit['best_query_template'] = str(b_q_t)
        else:
            dict_to_emit['estimated_precision'] = None
            dict_to_emit['estimated_recall'] = None
            dict_to_emit['best_query_template'] = None
            
        dict_to_emit['has_previous'] = 'temp_has_previous'# len(self.examples_buffer) >= 1
        if message:
            dict_to_emit['_message'] = message
        return dict_to_emit
    

        
