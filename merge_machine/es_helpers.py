#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 00:38:16 2017

@author: leo
"""

import itertools
import json

import unidecode


def my_unidecode(string):
    '''unidecode or return empty string'''
    if isinstance(string, str):
        return unidecode.unidecode(string)
    else:
        return ''


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

def _gen_body(query_template, row, must_filters={}, must_not_filters={}, num_results=3):
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
                                  s_q_t[2] + s_q_t[3]: {'query': _remove_words(row[s_q_t[1]].str.cat(sep=' '), must_filters.get(s_q_t[2], [])),
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
                                 } for field, values in must_not_filters.items()],
                       'filter': [{'match': {field + DEFAULT_MUST_FIELD: {'query': ' AND '.join(values)}} # TODO: french?
                                 } for field, values in must_filters.items()],
                    })               
                  }
           }
    return body


def _gen_bulk(index_name, search_templates, must, must_not, num_results, chunk_size=100):
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
        bulk_body += json.dumps({"index" : index_name}) + '\n'
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

def _bulk_search(es, index_name, all_query_templates, rows, must_filters, must_not_filters, num_results=3):
    '''
    Searches for the values in rows with all the search templates in 
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
        
        bulk_body_gen = _gen_bulk(index_name, [x[1] for x in search_templates], 
                                  must_filters, must_not_filters, num_results)
        responses = []
        for bulk_body, _ in bulk_body_gen:
            responses.extend(es.msearch(bulk_body)['responses']) #, index=index_name)
            
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
            raise Exception('Problem with elasticsearch: could not perform all queries in 10 trials')
            
    return og_search_templates, full_responses