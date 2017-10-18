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
import copy
import itertools

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd

from es_helpers import _bulk_search


es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
 


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
    
    index_name = params['index_name']
    query_template = params['query_template']
    must_filters = params.get('must', {})
    must_not_filters = params.get('must_not', {})
    threshold = params['thresh']
    exact_pairs = params.get('exact_pairs', [])
    non_matching_pairs = params.get('non_matching_pairs', [])
    
    exact_source_indexes = [x[0] for x in exact_pairs if x[1] is not None]
    exact_ref_indexes = [x[1] for x in exact_pairs if x[1] is not None]
    source_indexes = [x[0] for x in source.iterrows() if x [0] not in exact_source_indexes]
    
    def _is_match(f_r, threshold):
        return bool(f_r['hits']['hits']) and (f_r['hits']['max_score'] >= threshold)
    
    # Perform matching on non-exact pairs (not labelled)
    if source_indexes:
        rows = (x[1] for x in source.iterrows() if x[0] in source_indexes)
        all_search_templates, full_responses = _bulk_search(es, index_name, [query_template], rows, must_filters, must_not_filters, num_results=1)
        full_responses = [full_responses[i] for i in range(len(full_responses))] # Don't use items to preserve order

        # TODO: remove threshold condition
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
                                if (len(f_r['hits']['hits']) >= 2) and _is_match(f_r, threshold) \
                                else np.nan \
                                for f_r in full_responses], index=matches_in_ref.index)

        matches_in_ref.columns = [x + '__REF' for x in matches_in_ref.columns]
        matches_in_ref['__ID_REF'] = ref_id
        matches_in_ref['__CONFIDENCE'] = confidence    
        matches_in_ref['__GAP'] = confidence_gap
        matches_in_ref['__GAP_RATIO'] = confidence_gap / confidence

        # Put confidence to zero for user labelled negative pairs
        sel = [x in non_matching_pairs for x in zip(source_indexes, matches_in_ref.__ID_REF)]
        for col in ['__CONFIDENCE', '__GAP', '__GAP_RATIO']:
            matches_in_ref.loc[sel, '__CONFIDENCE'] = 0    
        
    else:
        matches_in_ref = pd.DataFrame()
        
    # Perform matching exact (labelled) pairs
    if exact_ref_indexes:
        full_responses = [es.get(index_name, ref_idx) for ref_idx in exact_ref_indexes]
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


    

        
