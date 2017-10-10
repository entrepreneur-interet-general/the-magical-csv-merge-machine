#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:43:40 2017

@author: m75380

Directions:
    - Parents do not include filters
    
    
TODO:

- ref_gen
- buffer_for_future_matches in same row when previous is called:
    overwrite ref_gen
- query_filter    
    
"""

import itertools
import random

from elasticsearch import Elasticsearch
import numpy as np

from es_helpers import _gen_body, _bulk_search


class SingleQueryTemplate():
    
    def __init__(self, bool_lvl, source_col, ref_col, analyzer_suffix, boost):
        self.bool_lvl = bool_lvl
        
        if isinstance(source_col, str):
            source_col = [source_col]
        if isinstance(ref_col, str):
            ref_col = [ref_col]
            
        self.source_col = tuple(sorted(source_col))
        self.ref_col = tuple(sorted(ref_col))
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
        try:
            return self.__hash
        except:
            self.__hash = hash((self.bool_lvl, tuple(sorted(self.source_col)), tuple(sorted(self.source_col)), 
                     self.analyzer_suffix, self.boost))
            return self.__hash
    
    def __eq__(self, other):
        return self._as_tuple() == other._as_tuple()
    

    def __gt__(self, other):
        return self._as_tuple() > other._as_tuple()

#    def __str__(self):
#        return self._as_tuple().__str__()
    
    def _as_tuple(self):
        return (self.bool_lvl, self.source_col, self.ref_col, self.analyzer_suffix, self.boost)
    
    #    def __eq__(self, other):
    #        if not isinstance(other, type(self)):
    #            return False
    #        return (self.bool_lvl, self.source_col, self.ref_col, self.analyzer_suffix, self.boost)


class CompoundQueryTemplate():
    '''Information regarding a query to be used in the labeller'''
    def __init__(self, single_query_templates_tuple):        
        self.musts = tuple(sorted([SingleQueryTemplate(*x) for x in single_query_templates_tuple if x[0] == 'must']))
        self.shoulds = tuple(sorted([SingleQueryTemplate(*x) for x in single_query_templates_tuple if x[0] == 'should']))
        
        assert self.musts
        
        self.core = self._core()
        self.parent_cores = self._parent_cores()

    def _core(self):
        '''Same as in SingleQueryTemplate'''
        cores = []
        for must in self.musts:
            cores.append(must._core())
        return tuple(sorted(cores))

    def __hash__(self):
        try:
            return self.__hash
        except:
            self.__hash = hash((q.__hash__() for q in self.musts + self.shoulds))    
            return self.__hash

    def __eq__(self, other):        
        bool_1 = self.musts == other.musts
        bool_2 = self.shoulds == other.shoulds
        
        return bool_1 and bool_2


#    def __str__(self):
#        return [x.__str__() for x in self.musts + self.shoulds].__str__()

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
        return tuple(x._as_tuple() for x in self.musts + self.shoulds)
     
        
class LabellerQuery(CompoundQueryTemplate):
    '''Extension of CompoundQueryTemplate which can also store history of labels'''
    
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        
        self.history_pairs = [] # [[(1,5), (1,2)], [(2,3), (2,5)], []]
        #self.steps = [] # Num rows labelled at which this pair was added (in case of later creation)
        self.first_scores = [] # ES scores of first result of queies
        self.has_match = []
        self.first_is_match = []
        
        self.thresh = None
        self.precision = None
        self.recall = None
        self.score = 0.1 # General score of the template based on precision and recall

    def add_pairs(self, pairs, first_score):
        '''Add pairs to history_pairs (pairs that were generated by this query)'''
        # TODO: add steps
        self.history_pairs.append(pairs)
        self.has_match.append(bool(pairs))
        self.first_scores.append(first_score)
        
    def add_labelled_pair(self, labelled_pair):
        ''' Update first_is_match with a labelled_pair (commes after add_pairs)'''
        
        assert len(self.first_scores) == len(self.history_pairs)
        
        # If row is forgotten or no matches were found in all queries, mark as unaplicable
        if (labelled_pair is None) or (labelled_pair[1] is None): 
            self.first_is_match.append(None) # None if does not apply
        else:
            if self.history_pairs[-1]:        
                self.first_is_match.append(labelled_pair==self.history_pairs[-1][0])
            else:
                self.first_is_match.append(False)
            
    def previous(self, num_labelled):
        ''' Go back to state when num_labelled were labelled'''
        # Compute idx to go back to
        # TODO: check that steps are ordered
        idx = num_labelled 
        #        idx = sum(x <= num_labelled for x in self.steps) - 1
        #        assert idx >= 0
        
        self.history_pairs = self.history_pairs[:idx+1] # TODO: do this ?
        # self.steps = self.steps[:idx+1]
        self.first_scores = self.first_scores[:idx+1]
        self.has_match = self.has_match[:idx]
        self.first_is_match = self.first_is_match[:idx]

        # Check whether or not to re-compute scores 


    def compute_metrics(self, t_p=0.95, t_r=0.3):
        ''' 
        Compute the optimal threshold and the associated metrics 
        
        INPUT:
            - t_p: target_precision
            - t_r: target_recall
            
        OUTPUT:
            - thresh: optimal threshold (#TODO: explain more)
            - precision
            - recall
            - ratio
        '''
        summaries = [{'score': s, 'first_is_match': fim, 'has_match': hm} \
                for (s, fim, hm) in zip(self.first_scores, self.first_is_match, self.has_match)]
        
        # Filter out relevent summaries only
        summaries = [summary for summary in summaries if summary['first_is_match'] is not None]
        
        #
        num_summaries = len(summaries)
    
        # Sort summaries by score / NB: 0 score deals with empty hits
        summaries = sorted(summaries, key=lambda x: x['score'], reverse=True)
        
        # score_vect = np.array([x['_score_first'] for x in sorted_summaries])
        is_first_vect = np.array([bool(x['first_is_match']) for x in summaries])
        
        # If no matches are present
        # TODO: eliminates things from start ?
        if sum(is_first_vect) == 0:
            self.thresh = 10**3
            self.precision = 0
            self.recall = 0
            self.score = 0 
            return self.thresh, self.precision, self.recall, self.score
     
        
        rolling_precision = is_first_vect.cumsum() / (np.arange(num_summaries) + 1)
        rolling_recall = is_first_vect.cumsum() / num_summaries
        
        # TODO: WHY is recall same as precision ?
        
        # Compute ratio
        _f_precision = lambda x: max(x - t_p, 0) + min(t_p*(x/t_p)**3, t_p)
        _f_recall = lambda x: max(x - t_r, 0) + min(t_p*(x/t_r)**3, t_r)
        a = np.fromiter((_f_precision(xi) for xi in rolling_precision), rolling_precision.dtype)
        b = np.fromiter((_f_recall(xi) for xi in rolling_recall), rolling_recall.dtype)
        rolling_ratio = a*b
    
        # Find best index for threshold
        MIN_OBSERVATIONS = 6 # minimal number of values on which to comupute threshold etc...
        idx = max(num_summaries - rolling_ratio[::-1].argmax() - 1, min(MIN_OBSERVATIONS, num_summaries-1))
        
        # TODO: added if // was is das ?
        if idx == len(summaries) - 1:
            thresh = 0.0001
        else:
            thresh = summaries[idx]['score']
        
        precision = rolling_precision[idx]
        recall = rolling_recall[idx]
        score = rolling_ratio[idx]
        
        self.thresh = thresh
        self.precision = precision
        self.recall = recall
        self.score = score
        
        return self.thresh, self.precision, self.recall, self.score


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

def _gen_all_query_template_tuples(match_cols, columns_to_index, bool_levels, 
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

    all_query_templates = list(filter(lambda query: 'must' in [s_q_t[0] for s_q_t in query], \
                                all_query_templates))
    return all_query_templates


"""
HOW TO USE:
    
labeller = Labeller()

for x in range(100):
    to_display = labeller.to_emit()
    labeller.update('y')

labeller.export_params()
"""
    
class Labeller():
    '''
    Labeller object that stores previous labels and generates new matches
    for the user to label
    
    0.1) Initiate queries/ metrics/ history
    0.2) Read first row
    
    1) Perform queries / update history of hits
    2) Generate pairs to propose (based on sorted queries)
    3...) Until row is over: User inputs label
    4) Update metrics and history 
    5) Sort queries
    6) Gen new row and back to 1)
    '''
    NUM_RESULTS = 3
    
    MAX_NUM_SAMPLES = 100
    
    VALID_ANSWERS = {'yes': 'y',
                 'y': 'y',
                 '1': 'y',
                 
                 'no': 'n',
                 'n': 'n',
                 '0': 'n',
                 
                 'uncertain': 'u',
                 'u': 'u',
                 
                 'forget_row': 'f',
                 'f': 'f',
                 
                 'previous': 'p',
                 'p': 'p',

                 'quit': 'q'             
                 }

    #min_precision_tab = [(20, 0.7), (10, 0.5), (5, 0.3)]
    #max_num_keys_tab = [(20, 10), (10, 50), (7, 200), (5, 500), (0, 1000)]
    #num_results_labelling = 3
    
    MAX_NUM_LEVELS = 3 # Number of match clauses
    BOOL_LEVELS = {'.integers': ['must', 'should'], 
                   '.city': ['must', 'should']}
    BOOST_LEVELS = [1]
    
    t_p = 0.92
    t_r = 0.3
    
    
    def __init__(self, source, ref_index_name, 
                 match_cols, columns_to_index, 
                 certain_column_matches=None, 
                 must={}, must_not={}):
        '''
        source: pandas data_frame
        ref_index_name: name of the Elasticsearch index used as reference
        '''
        
        self.source = source
        self.ref_index_name = ref_index_name
        self.match_cols = match_cols      
        self.columns_to_index = columns_to_index
        
        self.es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
        
        self.labelled_pairs = [] # Flat list of labelled pairs
        self.labels = [] # Flat list of labels
        self.num_rows_labelled = [] # Flat list: at given label, how many were labelled. NB: starts at 0, becomes 1 at first yes/forgotten, or when next_row
        self.labelled_pairs_match = [] # For each row, the resulting match: (A, B) / no-match: (A, None) or forgotten: None
         
        
        self.must_filters = must
        self.must_not_filters = must_not
        
           
        self._init_queries(match_cols, columns_to_index) # creates self.current_queries
        #self._init_history() # creates self.history
        self._init_source_gen() # creates self.source_gen
        
        self.current_source_idx = None
        self.current_ref_idx = None
        
        self.current_source_item = None
        self.current_ref_item = None
        
        self.current_es_score = None

        self._first_pair()
    
    def _fetch_source_item(self, source_idx):
        return self.source.loc[source_idx, :]
    
    def _fetch_ref_item(self, ref_idx):
        # TODO: look into batching this
        return self.es.get(self.ref_index_name, ref_idx)['_source'] 

    
    def _init_queries(self, match_cols, columns_to_index):
        """Generates initial query templates"""
        all_query_template_tuples = _gen_all_query_template_tuples(match_cols, 
                                                           columns_to_index, 
                                                           self.BOOL_LEVELS, 
                                                           self.BOOST_LEVELS, 
                                                           self.MAX_NUM_LEVELS)        
        self.current_queries = [LabellerQuery(q_t_t) for q_t_t in all_query_template_tuples]            

    #    def _init_history(self):
    #        """Generate history object"""
    #        self.history = dict()
    #        for query in self.current_queries:
    #            self.history[query] = {
    #                                    'returned_pairs': [], # list of list of pairs proposed (source, ref)
    #                                    'has_hit': [], # indicates if any pair was returned
    #                                    'is_match': [], # indicates if the best hit was a match
    #                                    'step': [] # indicates the step at which the pair was added
    #                                    }

    
    def _init_source_gen(self):
        def temp():
            sources_done = [x[0] for x in self.labelled_pairs_match if x is not None] # TODO: forgotten can be re-labelled
            for idx in random.sample(list(self.source.index), self.MAX_NUM_SAMPLES):
                if idx not in sources_done:
                    yield (idx, self._fetch_source_item(idx))
        self.source_gen = temp()
        
    def _init_ref_gen(self):
        '''Generator of results for the current source element'''
        def temp():
            for i, query in enumerate(self.current_queries):
                self.current_query_ranking = i
                self.current_query = query # TODO: for display only
                for pair in query.history_pairs[-1]: # TODO: this is only idx of results: get results
                    try:
                        assert pair[0] == self.current_source_idx
                    except:
                        import pdb; pdb.set_trace()
                    if pair not in self.labelled_pairs:
                        item = self.ref_id_to_data[pair[1]]['_source'] # TODO: get item if it is not in ref_id_to_data
                        es_score = self.ref_id_to_data[pair[1]]['_score']
                        yield pair[1], item, es_score
                    # TODO: check that source idx is same as in source_gen
        self.ref_gen = temp()        

    def _first_pair(self):
        """Initialiaze labeller"""
        self.current_source_idx, self.current_source_item = next(self.source_gen)
        # Fetch data for next row
        results = self.pruned_bulk_search(self.current_queries, 
                                self.current_source_item, self.NUM_RESULTS)
        self.add_results(results)
        
        self._init_ref_gen()     
        (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)

        
    def _bulk_search(self, queries_to_perform, row, num_results):
        # TODO: use self.current_queries instead ?
        
        # Transform
        queries_to_perform_tuple = [x._as_tuple() for x in queries_to_perform]
        
        search_templates, full_responses = _bulk_search(self.es, 
                                             self.ref_index_name, 
                                             queries_to_perform_tuple, 
                                             [row],
                                             self.must_filters, 
                                             self.must_not_filters, 
                                             num_results)
        
        assert [x[1][0] for x in search_templates] == queries_to_perform_tuple
        
        new_full_responses = [full_responses[i]['hits']['hits'] for (i, _) in search_templates]
        return new_full_responses
        
    def pruned_bulk_search(self, queries_to_perform, row, num_results):
        ''' 
        Performs a smart bulk request, by not searching for templates
        if restrictions of these templates already did not return any results
        '''
        
        results = {}
        core_has_match = dict()
        
        num_queries_performed = 0
        
        sorted_queries = sorted(queries_to_perform, key=lambda x: len(x.core)) # NB: groupby needs sorted 
        for size, group in itertools.groupby(sorted_queries, key=lambda x: len(x.core)):
            size_queries = sorted(group, key=lambda x: x.core)
            
            # 1) Fetch first of all unique cores            
            query_bulk = []
            for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
                # Only add core if all parents can have results
                core_queries = list(sub_group)
                first_query = core_queries[0]
                if all(core_has_match.get(parent_core, True) for parent_core in first_query.parent_cores):
                    query_bulk.append(first_query)        
                else:
                    core_has_match[core] = False
                    
            # Perform actual queries
            num_queries_performed += len(query_bulk)
            bulk_results = self._bulk_search(query_bulk, row, num_results)
            
            # Store results
            results.update(zip(query_bulk, bulk_results))
            for query, res in zip(query_bulk, bulk_results):
                core_has_match[query.core] = bool(res)
                
                
            # 2) Fetch queries when cores have results
            query_bulk = []
            for core, sub_group in itertools.groupby(size_queries, key=lambda x: x.core):
                if core_has_match[core]:
                    query_bulk.extend(list(sub_group))
                    
 
            # Perform actual queries
            num_queries_performed += len(query_bulk)
            bulk_results = self._bulk_search(query_bulk, row, num_results)
            
            # Store results
            results.update(zip(query_bulk, bulk_results))
            
        # Order responses
        to_return = [results.get(query, []) for query in queries_to_perform]
        
        if not sum(bool(x) for x in to_return):
            import pdb; pdb.set_trace()
        
        print('Num queries before pruning:', len(queries_to_perform))
        print('Num queries performed:', num_queries_performed)
        print('Non empty results:', sum(bool(x) for x in to_return))
        return to_return


    def _compute_metrics(self):
        '''Compute metrics (threshold, precision, recall, score) for each individual query'''
        for query in self.current_queries:
            query.compute_metrics(self.t_p, self.t_r)
            
    def _sort_queries(self):
        '''Sort queries by score (best first)'''
        self.current_queries = sorted(self.current_queries, key=lambda x: x.score, reverse=True)
                
    def _sorta_sort_queries(self):
        '''
        Alternate between random queries and sorted by best ES score (when 
        there are not enough data to compute real precision, recall, scores...)
        '''
        # Alternate between random and largest score
        queries = random.sample(self.current_queries, len(self.current_queries))
        
        a = queries[:int(len(queries)/2)]
        a = sorted(a, key=lambda x: x.first_scores[-1], reverse=True)
        
        if len(queries)%2 == 0:
            b = queries[int(len(queries)/2):]
            c = []
        else:
            b = queries[int(len(queries)/2):-1]
            c = [queries[-1]]
            
        # d_q = self._default_query()
        self.sorted_keys = [x for x in list(itertools.chain(*zip(a, b)))] + [c]    


    def previous(self):
        '''Goes to previous state'''
        # Re-place current values in todo-stack
        s_elem = (self.current_source_idx, self.current_source_item)
        r_elem = (self.current_ref_idx, self.current_ref_item, self.current_es_score)
        self.source_gen = itertools.chain([s_elem], self.source_gen)
        self.ref_gen = itertools.chain([r_elem], self.ref_gen)
        
        # TODO: test that previous is possible
        (self.current_source_idx, self.current_ref_idx) = self.labelled_pairs.pop()
        self.labels.pop()
        num_rows_labelled = self.num_rows_labelled.pop()
    
        # self.labelled_pairs_match = self.labelled_pairs_match[:num_rows_labelled] # For each row, the resulting match: (A, B) / no-match: (A, None) or forgotten: None
            
        self.current_source_item = self._fetch_source_item(self.current_source_idx)
        self.current_ref_item = self._fetch_ref_item(self.current_ref_idx)   
        
        self.current_es_score = None
        
        # Previous on all queries
        for query in self.current_queries:
            query.previous(num_rows_labelled)
            
            
        


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
                                                          self.MAX_NUM_SAMPLES))
        self.next_row = True
        

#    def add_pairs(self, pairs, first_score):
#        '''Add pairs to history_pairs (pairs that were generated by this query)'''
#        self.history_pairs.append(pairs)
#        self.has_match.append(bool(pairs))
#        self.first_scores.append(first_score)
        
    def add_labelled_pair(self, labelled_pair):
        for query in self.current_queries:
            query.add_labelled_pair(labelled_pair)
        self.labelled_pairs_match.append(labelled_pair)

    def add_results(self, results):
        '''Add results (ordered by current_queries) to each individual query'''
        
        # TODO: look into keeping more data than just this round
        self.ref_id_to_data = dict()
        
        for query, ref_result in zip(self.current_queries, results):
            if ref_result:
                score = ref_result[0]['_score']
            else:
                score = 0
            query.add_pairs([(self.current_source_idx, x['_id']) for x in ref_result], score)
            
            for res in ref_result:
                if res['_id'] not in self.ref_id_to_data:
                    self.ref_id_to_data[res['_id']] = res
            

    def update(self, user_input):
        '''
        Update query summaries and query_metrics and other variables based 
        on the user given label and the elasticsearch id of the reference item being labelled
        
        INPUT:
            - user_input: 
                + "y" or "1" or "yes": res_id is a match with self.idx
                + "n" or "0" or "no": res_id is not a match with self.idx
                + "u" or "uncertain": uncertain #TODO: is this no ?
                + "f" or "forget_row": uncertain #TODO: is this no ?
                + "p" or "previous": back to previous state
            - ref_id: Elasticsearch Id of the reference element being labelled
        '''
        MIN_NUM_KEYS = 9 # Number under which to expand
        EXPAND_FREQ = 9
        
        yes = self.VALID_ANSWERS[user_input] == 'y'
        no = self.VALID_ANSWERS[user_input] == 'n'
        uncertain = self.VALID_ANSWERS[user_input] == 'u'
        forget_row = self.VALID_ANSWERS[user_input] == 'f'
        use_previous = self.VALID_ANSWERS[user_input] == 'p'
        quit_ = self.VALID_ANSWERS[user_input] == 'q'
        
        assert yes + no + uncertain + forget_row + use_previous + quit_ == 1
    
        if quit_:
            raise RuntimeError('User quit labeller by typing "quit"')
    
        if use_previous:
            self.previous()
            return

        pair = (self.current_source_idx, self.current_ref_idx)
        
        self.labelled_pairs.append(pair)
        self.labels.append(user_input)

        print('in update')
        
        if yes:
            labelled_pair = pair
            next_row = True
        
        if no:
            next_row = False
            
        if uncertain:
            raise NotImplementedError('Uncertain is not yet implemented')
            
        if forget_row:
            labelled_pair = None
            next_row = True
            
        if not next_row:
            # Try to get next label, otherwise jump        
            try:
                (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)

                # If iterator runs out: next_row = True
                if self.num_rows_labelled:
                    self.num_rows_labelled.append(self.num_rows_labelled[-1] + 1)
                else:
                    self.num_rows_labelled = [0]
                                
            except StopIteration:
                # If no match was found
                labelled_pair = (self.current_source_idx, None)
                next_row = True
        
        # NB: not using "else" because there is a chance for next_row to chang in previous "if"    
        if next_row:
            # Add rows_labelled_counts
            if self.num_rows_labelled:
                self.num_rows_labelled.append(self.num_rows_labelled[-1] + 1)
            else:
                self.num_rows_labelled = [1] # TODO: check this ?

            # Re-score and sort metrics
            self.add_labelled_pair(labelled_pair)
            
            # Update metrics
            if labelled_pair is not None:
                self._compute_metrics()
                if True: # TODO: use sort_queries
                    self._sorta_sort_queries()
                    self._sort_queries()
                self._filter_queries() # TODO: implement this
            
            # Generate next row
            self.current_source_idx, self.current_source_item = next(self.source_gen)
            
            # Fetch data for next row
            results = self.pruned_bulk_search(self.current_queries, 
                                    self.current_source_item, self.NUM_RESULTS)
            self.add_results(results)
            
            self._init_ref_gen()
            
            (self.current_ref_idx, self.current_ref_item, self.current_es_score) = next(self.ref_gen)


    def _re_score_history(self):        
        # TODO: choose between keeping old or full re-initialisation
        print('WARNING: re-scoring history')
        
        self._init_queries(self.match_cols, self.columns_to_index)
        
        og_labelled_pairs = list(self.labelled_pairs_match)
        self.labelled_pairs_match = []
        for labelled_pair in og_labelled_pairs:
            (source_idx, ref_idx) = labelled_pair
            if (labelled_pair is not None) and (ref_idx is not None): # TODO: not re_labelling Nones
                self.current_source_idx = source_idx
                self.current_ref_idx = ref_idx
                
                self.current_source_item = self._fetch_source_item(source_idx)
                self.current_ref_item = self._fetch_ref_item(ref_idx)
                
                self.current_es_score = None
                
                # Fetch data for next row
                results = self.pruned_bulk_search(self.current_queries, 
                                        self.current_source_item, 1) #self.NUM_RESULTS)
                self.add_results(results)
                
                self.add_labelled_pair(labelled_pair)


        # Re-score metrics
        self._compute_metrics()
        if True: # TODO: use sort_queries
            self._sorta_sort_queries()
            self._sort_queries()
        
        self._first_pair()


    def _filter_queries(self):
        pass



            
        # TODO: num_rows_labelled is not good since it can not change on new label
        
        # TODO: deal with expansion
        
        #        try:
        #            self.last_expanded
        #        except:
        #            self.last_expanded = None
        #        if ((len(self.sorted_keys) < MIN_NUM_KEYS) \
        #            or((self.num_rows_labelled+1) % EXPAND_FREQ==0)) \
        #            and (self.last_expanded != self.idx):
        #            self.sorted_keys = _expand_by_boost(self.sorted_keys)
        #            self.re_score_history()    
        #            self.last_expanded = self.idx

        
        
    def answer_is_valid(self, user_input):
        '''Check if the user input is valid''' # DONE
        return user_input in self.VALID_ANSWERS

    def export_best_params(self, p):
        '''Returns a dictionnary with the best parameters (input for es_linker)''' # DONE
        params = dict()
        params['index_name'] = self.ref_table_name
        params['query_template'] = None # TODO:
        params['must'] = self.must_filters
        params['must_not'] = self.must_not_filters
        params['thresh'] = 0 
        params['best_thresh'] = self.metrics[params['query_template']]['thresh']
        
        params['exact_pairs'] = [p for (p, l) in zip(self.labelled_pairs, self.labels) if l == 'y']
        params['non_matching_pairs'] = [p for (p, l) in zip(self.labelled_pairs, self.labels) if l == 'n']
        params['forgotten_pairs'] = [p for (p, l) in zip(self.labelled_pairs, self.labels) if l == 'f']
        
        return params
    
    def write_training(self, file_path): # DONE     
        params = self.export_best_params()
        encoder = MyEncoder()
        with open(file_path, 'w') as w:
            w.write(encoder.encode(params))
    
    def update_musts(self, must_filters, must_not_filters): # DONE
        if (not isinstance(must_filters, dict)) or (not isinstance(must_not_filters, dict)):
            raise ValueError('Variables "must" and "must_not" should be dicts' \
                'with keys being column names and values a list of strings')
        self.must_filters = must_filters
        self.must_not_filters = must_not_filters
        self._re_score_history()
        
    def _best_query_template(self):
        """Return query template with the best score (ratio)"""
        if self.query_metrics:
            return sorted(self.metrics.keys(), key=lambda x: \
                          self.metrics[x]['ratio'], reverse=True)[0]
        else:
            return None
 
    def to_emit(self):
        '''Creates a dict to be sent to the template #TODO: fix this''' # DONE-ISH
        dict_to_emit = dict()

        # Info on labeller
        dict_to_emit['t_p'] = self.t_p
        dict_to_emit['t_r'] = self.t_r
        dict_to_emit['has_previous'] = None # TODO: 
        #        dict_to_emit['num_proposed_source'] = str(self.num_rows_proposed_source)
        #        dict_to_emit['num_proposed_ref'] = str(sum(self.num_rows_proposed_ref.values()))
        #        dict_to_emit['num_labelled'] = str(self.num_rows_labelled)
        
        # Info on current query
        current_query = self.current_query
        dict_to_emit['query'] = current_query._as_tuple()
        dict_to_emit['query_ranking'] = self.current_query_ranking
        dict_to_emit['estimated_precision'] = current_query.precision # TODO: 
        dict_to_emit['estimated_recall'] = current_query.recall # TODO: 
        dict_to_emit['estimated_score'] = current_query.score # TODO:   
        dict_to_emit['thresh'] = current_query.thresh        

        # Info on pair
        dict_to_emit['source_idx'] = self.current_source_idx
        dict_to_emit['ref_idx'] = self.current_ref_idx
        
        dict_to_emit['source_item'] = self.current_source_item
        dict_to_emit['ref_item'] = self.current_ref_item
        
        dict_to_emit['es_score'] = self.current_es_score
        
        if (self.current_es_score is not None) and (current_query.thresh is not None):
            dict_to_emit['estimated_is_match'] = self.current_es_score >= current_query.thresh
        else:
            dict_to_emit['estimated_is_match'] = None       
        
        return dict_to_emit
    
    
    def print_emit(self):
        '''Print current state of labeller'''
        dict_to_emit = self.to_emit()
        
        print('\n' + '*'*50)
        print('({0}): {1}'.format(dict_to_emit['query_ranking'], dict_to_emit['query']))
        print('Precision: {0}; Recall: {1}; Score: {2}'.format(
                                          dict_to_emit['estimated_precision'],
                                          dict_to_emit['estimated_recall'],
                                          dict_to_emit['estimated_score']))
    
        print('ES score: {0}; Thresh: {1}; Is match: {2}'.format(dict_to_emit['es_score'],
                  dict_to_emit['thresh'], dict_to_emit['estimated_is_match']))
    
        print('\n(S): {0}'.format(dict_to_emit['source_idx']))
        print('(R): {0}'.format(dict_to_emit['ref_idx']))

        for match in self.match_cols:
            print('\n')
            source_cols = match['source']
            if isinstance(source_cols, str):
                source_cols = [source_cols]
            ref_cols = match['ref']
            if isinstance(ref_cols, str):
                ref_cols = [ref_cols]
                
            try:
                for source_col in source_cols:
                    print('(S): {0} -> {1}'.format(source_col, dict_to_emit['source_item'][source_col]))
                    
                for ref_col in ref_cols:
                    print('(R): {0} -> {1}'.format(ref_col, dict_to_emit['ref_item'][ref_col]))
            except:
                import pdb; pdb.set_trace()

    
    def console_input(self):
        ''' '''
        self.print_emit()
        return input('\n > ')
        

        
        