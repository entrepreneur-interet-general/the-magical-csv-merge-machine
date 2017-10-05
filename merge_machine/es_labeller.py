#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:43:40 2017

@author: m75380

Directions:
    - Parents do not include filters
"""

import itertools

from elasticsearch import Elasticsearch
import numpy as np

from es_helpers import _gen_body, _bulk_search


class SingleQueryTemplate():
    
    def __init__(self, bool_lvl, source_col, ref_col, analyzer_suffix, boost):
        self.bool_lvl = bool_lvl
        self.source_col = source_col
        self.ref_col = ref_col
        self.analyzer_suffix = analyzer_suffix
        self.boost = boost

    def _core(self):
        '''
        Return the minimal compononent that guarantees equivalence of the claim:
        "this query has results"    
        '''
        return (self.source_col, self.ref_col, self.analyzer_suffix)
        
    def __hash__(self):
        # TODO: join is not clean
        return hash(self.bool_lvl, '/./'.join(self.source_col), '/./'.join(self.ref_col), self.analyzer_suffix, self.boost)
    
    def _as_tuple(self):
        return (self.bool_lvl, self.source_col, self.ref_col, self.analyzer_suffix, self.boost)
    
    #    def __eq__(self, other):
    #        if not isinstance(other, type(self)):
    #            return False
    #        return (self.bool_lvl, self.source_col, self.ref_col, self.analyzer_suffix, self.boost)


class CompoundQueryTemplate():
    '''Information regarding a query to be used in the labeller'''
    def __init__(self, single_query_templates):        
        self.musts = [x for x in single_query_templates if x[0] == 'must']
        self.shoulds = [x for x in single_query_templates if x[0] == 'should']
        
        assert self.musts
        
        self.core = self._core()
        self.parent_cores = self._parent_cores()


    def _core(self):
        '''Same as in SingleQueryTemplate'''
        cores = []
        for must in self.musts:
            cores.append(must._core())
        return tuple(sorted(res))

    def __hash__(self):
        return hash((q.__hash__() for q in self.musts + self.shoulds))    

    def _parent_cores(self):
        '''Returns the core of all possible parents'''
        if len(self.core) == 1:
            return []
        else:
            to_return = []
            for i in range(1, len(self.core)-1):
                to_return.extend(list(itertools.combinations(self.core, i)))
            return to_return

        #    def add_returned_hits(self, returned_hits):
        #        self.returned_pairs_hist.append(returned_hits)
        #        
        #    def remove_last_returned_hits(self, returned_hits):
        #        '''Remove last element from returned hits'''
        #        if self.returned_pairs:
        #            self.returned_pairs_hist.pop()
        
    
    
    def _gen_body(self, row, must_filter={}, must_not_filter={}, num_results=3):
        '''
        Generate the string to pass to Elastic search for it to execute query
        
        INPUT:
            - row: pandas.Series from the source object
            - must: terms to filter by field (AND: will include ONLY IF ALL are in text)
            - must_not: terms to exclude by field from search (OR: will exclude if ANY is found)
            - num_results: Max number of results for the query
        
        OUTPUT:
            - body: the query as string
        '''
        body = _gen_body(self.must + self.should, row, must_filter, must_not_filter, num_results)
        return body
        
    def _as_tuple(self):
        tuple(x._as_tuple for x in self.musts + self.shoulds)
        
        
class Labeller():
    '''
    Labeller object that stores previous labels and generates new matches
    for the user to label
    
    
    '''
    
    def __init__(self, ref_index_name):
        
        self.ref_index_name = ref_index_name
        
        self.es = Elasticsearch()
        
        self.labelled_pairs = [] # Flat list of labelled pairs
        self.labels = [] # Flat list of labels

        self._init_queries() # creates self.current_queries
        self._init_metrics() # creates self.metrics
        self._init_history() # creates self.history
    
    def _init_queries(self, ):
        """Generates initial query templates"""
        self.current_queries = []
        
    def _init_metrics(self):
        """Generate metrics object"""
        self.metrics = dict()
        for query in self.current_queries:
            self.metrics[query] = {'precision': None,
                                    'recall': None,
                                    'score': None}
    
    def _init_history(self):
        """Generate history object"""
        self.history = dict()
        for query in self.current_queries:
            self.history[query] = {
                                    'returned_pairs': [], # list of list of pairs proposed (source, ref)
                                    'has_hit': [], # indicates if any pair was returned
                                    'is_match': [], # indicates if the best hit was a match
                                    'step': [] # indicates the step at which the pair was added
                                    }
        
    def _bulk_search(self, queries_to_perform, row):
        # TODO: use self.current_queries instead ?
        NUM_RESULTS = 3
        
        # Transform
        queries_to_perform_tuple = [x.as_tuple() for x in queries_to_perform]
        search_template, res = _bulk_search(self.es, 
                                             self.ref_index_names, 
                                             queries_to_perform_tuple, 
                                             [row],
                                             self.must_filters, 
                                             self.must_not_filters, 
                                             NUM_RESULTS)
        
        [(0, (query, row)), ...]; full_responses is {0: res, 1: res, ...}
        
    def pruned_bulk_search(self, queries_to_perform, row):
        ''' 
        Performs a smart bulk request, by not searching for templates
        if restrictions of these templates already did not return any results
        '''
        
        queries_performed = []
        results = {}
        
        core_has_match = dict()
        
        sorted_queries = sorted(queries_to_perform, key=lambda x: len(x.core)) # NB: groupby needs sorted 
        for size, group in itertools.groupby(sorted_queries, key=lambda x: len(x.core)):
            size_queries = sorted(group, lambda x: x.core)
            
            # 1) Fetch first of all unique cores            
            query_bulk = []
            for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
                # Only add core if all parents can have results
                core_queries = list(sub_group)
                first_query = core_queries[0]
                if all(core_has_match.get(parent_core, True) for parent_core in first_query.core_parents):
                    query_bulk.append(first_query)        
                else:
                    core_has_match[core] = False
                    
            # Perform actual queries
            bulk_results = self._bulk_request(query_bulk, assert False)
            
            # Store results
            results.update(zip(query_bulk, bulk_results))
            for query, res in zip(query_bulk, bulk_results):
                core_has_match[query.core] = bool(res)
                
                
            # 2) Fetch queries when cores have 
            query_bulk = []
            for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
                if core_has_match[core]:
                    query_bulk.extend(list(sub_group))
 
            # Perform actual queries
            bulk_results = self._bulk_search(query_bulk)
            
            # Store results
            results.update(zip(query_bulk, bulk_results))
            
        # Order responses
        to_return = [results.get(query, []) for query in queries_to_perform]        
        return to_return
            
        
        
    def re_score_history(self):
        '''Use this After a must and or must_not update, '''
        
        
        