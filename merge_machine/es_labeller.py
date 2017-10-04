#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:43:40 2017

@author: m75380

Directions:
    - Parents do not include filters
"""

import itertools

import numpy as np


class SingleQueryTemplate():
    
    def __init__(self, bool_lvl, source_col, ref_col, analyzer_suffix, boost):
        self.bool_lvl = bool_lvl
        self.source_col = source_col
        self.ref_col = ref_col
        self.analyzer_suffix = analyzer_suffix
        self.boost = boost
        
    def __hash__(self):
        return (self.bool_lvl, self.source_col, self.ref_col, self.analyzer_suffix, self.boost)
    
    def _core(self):
        '''
        Return the minimal compononent that guarantees equivalence of the claim:
        "this query has results"    
        '''
        return (self.source_col, self.ref_col, self.analyzer_suffix)


class CompoundQueryTemplate():
    '''Information regarding a query to be used in the labeller'''
    def __init__(self, query_templates):        
        self.musts = [x for x in query_templates if x[0] == 'must']
        self.shoulds = [x for x in query_templates if x[0] == 'should']
        
        assert self.musts
        
        self.core = self._core()
        self.core_parents = self._core_parents()

    def _core(self):
        '''Same as in SingleQueryTemplate'''
        cores = []
        for must in self.musts:
            cores.append(must._core())
        return tuple(sorted(res))

    def _core_parents(self):
        '''Returns the core of all possible parents'''
        if len(self.core) == 1:
            return []
        else:
            return list(itertools.combinations(self.core, len(self.core) - 1))

        #    def add_returned_hits(self, returned_hits):
        #        self.returned_pairs_hist.append(returned_hits)
        #        
        #    def remove_last_returned_hits(self, returned_hits):
        #        '''Remove last element from returned hits'''
        #        if self.returned_pairs:
        #            self.returned_pairs_hist.pop()
        
    
    
    def _gen_body(self, row, num_results=3):
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
        

class Labeller():
    '''
    Labeller object that stores previous labels and generates new matches
    for the user to label
    '''
    
    def __init__(self):
        self.labelled_pairs = [] # Flat list of labelled pairs
        self.labels = [] # Flat list of labels
        
        self.returned_pairs_hist = [] # History: list of list of pairs proposed (source, ref)
        self.has_hit_hist = [] # History: Indicates if any pair was returned
        self.is_match_hist = [] # History: Indicates if the best hit was a match
        
        self.precision = None
        self.recall = None
        self.score = None
                
        
    
    def perform_request(self):
        ''''''
        
        
    def re_score_history(self):
        '''Use this After a must and or must_not update, '''
        
        
        