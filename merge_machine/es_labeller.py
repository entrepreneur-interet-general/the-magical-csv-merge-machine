#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:43:40 2017

@author: m75380

Directions:
    - Parents do not include filters
"""

import itertools
import random

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
        return tuple(sorted(cores))

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
    
    0.1) Initiate queries/ metrics/ history
    0.2) Read first row
    
    1) Perform queries / update history of hits
    2) Generate pairs to propose (based on sorted queries)
    3...) Until row is over: User inputs label
    4) Update metrics and history 
    5) Sort queries
    6) Gen new row and back to 1)
    '''
    
    max_num_samples = 100
    
    def __init__(self, source, ref_index_name):
        '''
        source: pandas data_frame
        ref_index_name: name of the Elasticsearch index used as reference
        '''
        self.source = source
        self.ref_index_name = ref_index_name
        
        self.source_row_gen = self._init_row_gen()
        self.es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
        
        self.labelled_pairs = [] # Flat list of labelled pairs
        self.labels = [] # Flat list of labels
        self.num_rows_labelled = [] # Flat list: at given label, how many were labelled
        
        self.labelled_pairs_match = [] # For each row, the resulting match: (A, B) / no-match: (A, None) or forgotten: None
            
        self._init_queries() # creates self.current_queries
        self._init_metrics() # creates self.metrics
        self._init_history() # creates self.history
        
        self.current_source_idx = None
        self.current_ref_idx = None
        
        self.current_source_item = None
        self.current_ref_item = None
    
    def _init_queries(self, ):
        """Generates initial query templates"""
        self.current_queries = []
        
#    def _init_metrics(self):
#        """Generate metrics object"""
#        self.metrics = dict()
#        for query in self.current_queries:
#            self.metrics[query] = {'precision': None,
#                                    'recall': None,
#                                    'score': None}
    
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
    
    def _compute_metrics(self):
        

            
        for query in self.current_queries:
            returned_pairs = self.history[query]['returned_pairs']
            steps = self.history[query]['step']
            
                    for pair_match in self.labelled_pairs_match:
            if pair_match is not None:
            for step, pair in returned_pairs
                
    
    def _init_row_gen(self):
        for idx in random.sample(self.source.index, self.max_num_samples):
            yield (idx, self.source.loc[idx, :])
        
        
    def _bulk_search(self, queries_to_perform, row, num_results):
        # TODO: use self.current_queries instead ?
        
        # Transform
        queries_to_perform_tuple = [x.as_tuple() for x in queries_to_perform]
        search_templates, full_responses = _bulk_search(self.es, 
                                             self.ref_index_names, 
                                             queries_to_perform_tuple, 
                                             [row],
                                             self.must_filters, 
                                             self.must_not_filters, 
                                             num_results)
        
        assert [x[1][0] for x in search_templates] == queries_to_perform_tuple
        return [x['hits']['hits'] for x in full_responses]
        
    def pruned_bulk_search(self, queries_to_perform, row, num_results):
        ''' 
        Performs a smart bulk request, by not searching for templates
        if restrictions of these templates already did not return any results
        '''
        
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
            bulk_results = self._bulk_search(query_bulk, row, num_results)
            
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
            bulk_results = self._bulk_search(query_bulk, row, num_results)
            
            # Store results
            results.update(zip(query_bulk, bulk_results))
            
        # Order responses
        to_return = [results.get(query, []) for query in queries_to_perform]        
        return to_return






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
        pass


    def _sort_queries(self):
        '''
        Update sorted_keys, that determin the order in which samples are shown
        to the user        
        '''
        
        # Sort keys by score or most promising
        if self.num_rows_labelled <= 3:
            # Alternate between random and largest score
            sorted_keys_1 = random.sample(list(self.full_responses.keys()), len(self.full_responses.keys()))
            sorted_keys_2 = sorted(self.full_responses.keys(), key=lambda x: \
                                   self.full_responses[x]['hits'].get('max_score') or 0, reverse=True)
            
            d_q = self._default_query()
            self.sorted_keys =  [d_q] \
                        + [x for x in list(itertools.chain(*zip(sorted_keys_2, sorted_keys_1))) if x != d_q]
            # TODO: redundency with first
        else:
            # Sort by ratio but label by precision ?
            self.sorted_keys = sorted(self.full_responses.keys(), key=lambda x: self.query_metrics[x]['precision'], reverse=True)
            
            
            l1 = len(self.sorted_keys)
            # Remove queries if precision is too low (thresh depends on number of labels)
            self.sorted_keys = list(filter(lambda x: self.query_metrics[x]['precision'] \
                                      >= self._min_precision(), self.sorted_keys))
            l2 = len(self.sorted_keys)
            print('min_precision: removed {0} queries; {1} left'.format(l1-l2, l2))
            
            # Remove queries according to max number of keys
            self.sorted_keys = self.sorted_keys[:self._max_num_keys()]
            l3 = len(self.sorted_keys)
            print('max_num_keys: removed {0} queries; {1} left'.format(l2-l3, l3))
                
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

        
    def _new_label_for_row(self, full_responses, sorted_keys, num_results):
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
            results = full_responses[key]['hits']['hits']
            
            if len(results):
                print('\nkey (ex {0}/{1}): {2}'.format(i, num_keys, key))
                print('Num hits for this key: ', len(results)) 
            else:
                print('\nkey ({0}/{1}) has no results...'.format(i, num_keys))
                
            try:
                print('precision: ', self.query_metrics[key]['precision'])
                print('recall: ', self.query_metrics[key]['recall'])
            except:
                print('no precision/recall to display...')                
                
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
            
            print('\n' + '*'*40 + '\n', 'in new_label / in self.next_row / len sorted_keys: {0} / row_idx: {1}'.format(len(self.sorted_keys), self.idx))
            all_search_templates, tmp_full_responses = perform_queries(self.ref_table_name, self.sorted_keys, [row], self.must, self.must_not, self.num_results_labelling)
            self.full_responses = {all_search_templates[i][1][0]: values for i, values in tmp_full_responses.items()}
            print('LEN OF FULL_RESPONSES (number of queries):', len(self.full_responses))
            # import pdb; pdb.set_trace()
            self._sort_queries()
            
            print('BEST 10 KEYS:')
            for key in self.sorted_keys[:10]:
                print('\n', key)
                metrics = self.query_metrics.get(key, {})
                print(' > precision:', metrics.get('precision')) 
                print(' > recall:', metrics.get('recall'))
                print(' > ratio:', metrics.get('ratio'))
                
            self.label_row_gen = self._new_label_for_row(self.full_responses, 
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
            #            self.source_item = None
            #            self.ref_item = None
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
                + "y" or "1" or "yes": res_id is a match with self.idx
                + "n" or "0" or "no": res_id is not a match with self.idx
                + "u" or "uncertain": uncertain #TODO: is this no ?
                + "f" or "forget_row": uncertain #TODO: is this no ?
                + "p" or "previous": back to previous state
            - ref_id: Elasticsearch Id of the reference element being labelled
        '''
        MIN_NUM_KEYS = 9 # Number under which to expand
        EXPAND_FREQ = 9
        
        yes = user_input in ['y', '1', 'yes']
        no = user_input in ['n', '0', 'no']
        uncertain = user_input in ['u', 'uncertain']
        forget_row = user_input in ['f', 'forget_row']
        use_previous = user_input in ['p', 'previous']
        
        assert yes + no + uncertain + forget_row + use_previous == 1
        
        if use_previous:
            self.previous()
            return

        print('in update')
        if yes:
            if (self.pairs) and (self.pairs[-1][0] == self.idx):
                self.pairs.pop()
            self.pairs.append((self.idx, ref_id))

            self._update_query_summaries(ref_id)
            self.num_rows_labelled += 1
            self.next_row = True
        
        if no:
            self.pairs_not_match[self.idx].append(ref_id)
            
            if (not self.pairs) or (self.pairs[-1][0] != self.idx):
                self.pairs.append((self.idx, None))
        
        if uncertain:
            raise NotImplementedError('Uncertain is not yet implemented')
            
        if forget_row:
            self.next_row = True
            
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

        self._init_metrics() # re-initialisez self.metrics

        
        # TODO: temporary sol: put all in bulk
        for pair in self.pairs:
            # TODO: get row somehow
            all_search_templates, self.full_responses = self.pruned_bulk_search(
                                                                queries_to_perform,
                                                                row, 
                                                                1)
            # TODO: update metrics

        self._sort_queries()

        if not self.sorted_keys:
            raise ValueError('No keys in self.sorted_keys')
        
    def answer_is_valid(self, user_input):
        '''Check if the user input is valid''' # DONE
        valid_responses = {'y', '1', 'yes',
                           'n', '0', 'no',
                           'u', 'uncertain', 
                           'f', 'forget_row',
                           'p', 'previous'}
        return user_input in valid_responses

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
        self.re_score_history()
        
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
        # Info on pair
        dict_to_emit['source_idx'] = self.current_source_idx
        dict_to_emit['ref_idx'] = self.current_ref_idx
        
        dict_to_emit['source_item'] = self.current_source_item
        dict_to_emit['ref_item'] = self.current_ref_item
        
        # Info on past labelling
        dict_to_emit['num_proposed_source'] = str(self.num_rows_proposed_source)
        dict_to_emit['num_proposed_ref'] = str(sum(self.num_rows_proposed_ref.values()))
        dict_to_emit['num_labelled'] = str(self.num_rows_labelled)
        dict_to_emit['t_p'] = self.t_p
        dict_to_emit['t_r'] = self.t_r
        
        # Info on current performence
        assert False
        dict_to_emit['estimated_precision'] = None # TODO: 
        dict_to_emit['estimated_recall'] = None # TODO: 
        dict_to_emit['best_query_template'] = None # TODO: 
            
        dict_to_emit['has_previous'] = None # TODO: 
        
        return dict_to_emit
    

        

        
        