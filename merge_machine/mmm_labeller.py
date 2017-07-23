#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:18:51 2017

@author: leo
"""

from future.utils import viewitems, viewvalues

from collections import defaultdict, deque
import functools
import itertools
import json
import os
from string import punctuation
import random

import dedupe
from dedupe import blocking, predicates, sampling
from highered import CRFEditDistance
import numpy as np
import pandas as pd
import pprint
import unidecode



def pd_pre_process(series, remove_punctuation=False):
    '''Applies pre-processing to series using builtin pandas.str'''
    series = series.str.replace(' +', ' ')
    series = series.str.replace('\n', ' ')
    if remove_punctuation:
        for punc in punctuation:
            series = series.str.replace(punc, ' ')
    series = series.str.strip(' \"\'').str.lower()
    
    sel = series.notnull()
    series[sel] = series[sel].apply(lambda x: unidecode.unidecode(x))
    
    series = series.replace('', np.nan)
    return series 

def interleave(*iterables) :
    return itertools.chain.from_iterable(zip(*iterables))

def sort_pair(a, b) :
    if a > b :
        return (b, a)
    else :
        return (a, b)

def blockedSample(sampler, sample_size, my_predicates, *args) :
    
    blocked_sample = set()
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and my_predicates :
        random.shuffle(my_predicates)

        new_sample = sampler(remaining_sample, # change here
                             my_predicates,
                             *args)

        filtered_sample = ([(predicate, pair) for pair in subsample] for (predicate, subsample) 
                           in new_sample if subsample)
        
        blocked_sample.update(itertools.chain.from_iterable(filtered_sample))

        growth = len(blocked_sample) - previous_sample_size
        growth_rate = growth/remaining_sample

        remaining_sample = sample_size - len(blocked_sample)
        previous_sample_size = len(blocked_sample)

        if growth_rate < 0.001 :
            print("%s blocked samples were requested, "
                          "but only able to sample %s"
                          % (sample_size, len(blocked_sample)))
            break
        
        my_predicates = [pred for pred, pred_sample 
                      in zip(my_predicates, new_sample)
                      if pred_sample or pred_sample is None]
        
    return blocked_sample

def linkSamplePredicate(subsample_size, predicate, items1, items2) :
    sample = []

    predicate_function = predicate.func
    field = predicate.field

    red = defaultdict(list)
    blue = defaultdict(list)

    for i, (index, record) in enumerate(interleave(items1, items2)):
        if i == 20000:
            if min(len(red), len(blue)) + len(sample) < 10 :
                return sample

        column = record[field]
        if not column :
            red, blue = blue, red
            continue

        block_keys = predicate_function(column)
        for block_key in block_keys:
            if blue.get(block_key):
                pair = sort_pair(blue[block_key].pop(), index)
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break
                else :
                    return sample
            else:
                red[block_key].append(index)

        red, blue = blue, red

    for index, record in itertools.islice(items2, len(items1)) :
        column = record[field]
        if not column :
            continue

        block_keys = predicate_function(column)
        for block_key in block_keys:
            if red.get(block_key):
                pair = sort_pair(red[block_key].pop(), index)
                sample.append(pair)

                subsample_size -= 1
                if subsample_size :
                    break
                else :
                    return sample

    return sample

def evenSplits(total_size, num_splits) :
    avg = total_size/num_splits
    split = 0
    for _ in range(num_splits) :
        split += avg - int(split)
        yield int(split)

def subsample(total_size, my_predicates) :
    splits = evenSplits(total_size, len(my_predicates))
    for split, predicate in zip(splits, my_predicates) :
        yield split, predicate

def linkSamplePredicates(sample_size, my_predicates, items1, items2) :
    n_1 = len(items1)
    n_2 = len(items2)

    for subsample_size, predicate in subsample(sample_size, my_predicates) :
        
        if not subsample_size :
            yield predicate, None # change here
            continue

        try:
            items1.rotate(random.randrange(n_1))
            items2.rotate(random.randrange(n_2))
        except ValueError :
            raise ValueError("Empty itemset.")

        try :
            items1.reverse()
            items2.reverse()
        except AttributeError :
            items1 = deque(reversed(items1))
            items2 = deque(reversed(items2))

        yield predicate, linkSamplePredicate(subsample_size, predicate, items1, items2) # change here



def cover(blocker, pairs, compound_length) : # pragma: no cover
    cover = coveredPairs(blocker.predicates, pairs)
    cover = compound(cover, compound_length)
    cover = remaining_cover(cover)
    return cover

def coveredPairs(my_predicates, pairs) :
    cover = {}
        
    for predicate in my_predicates :
        cover[predicate] = {i for i, (record_1, record_2)
                            in enumerate(pairs)
                            if (set(predicate(record_1)) &
                                set(predicate(record_2)))}
    return cover

def compound(cover, compound_length) :
    simple_predicates = sorted(cover, key=str)
    CP = predicates.CompoundPredicate

    for i in range(2, compound_length+1) :
        compound_predicates = itertools.combinations(simple_predicates, i)
                                                             
        for compound_predicate in compound_predicates :
            a, b = compound_predicate[:-1], compound_predicate[-1]
            if len(a) == 1 :
                a = a[0]

            if a in cover:
                compound_cover = cover[a] & cover[b]
                if compound_cover:
                    cover[CP(compound_predicate)] = compound_cover

    return cover

def remaining_cover(coverage, covered=set()):
    remaining = {}
    for predicate, uncovered in viewitems(coverage):
        still_uncovered = uncovered - covered
        if still_uncovered:
            if still_uncovered == uncovered:
                remaining[predicate] = uncovered
            else:
                remaining[predicate] = still_uncovered

    return remaining

def unroll(matches) : # pragma: no cover
    return unique((record for pair in matches for record in pair))

def unique(seq):
    """Return the unique elements of a collection even if those elements are
       unhashable and unsortable, like dicts and sets"""
    cleaned = []
    for each in seq:
        if each not in cleaned:
            cleaned.append(each)
    return cleaned

def get_col(predicate):
    '''Returns the name of the column associated to the predicate'''
    return predicate.__name__[1:-1].split(',', 1)[-1].strip()


linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) 

dir_path = 'local_test_data'

source = pd.read_csv(os.path.join(dir_path, 'source.csv'), dtype=str)
ref = pd.read_csv(os.path.join(dir_path, 'ref.csv'), dtype=str)

match_cols = [{'source': 'departement', 'ref': 'departement'},
              {'source': 'commune', 'ref': 'localite_acheminement_uai'},
              {'source': 'lycees_sources', 'ref': 'full_name'}]

source_cols = [x['source'] for x in match_cols]
ref_cols = [x['ref'] for x in match_cols]


temp_match_cols = {x['source']: x['ref'] for x in match_cols}


# Replace column_names in source by those in ref
source.columns = [temp_match_cols.get(x, x) for x in source.columns]

fields = [{'crf': True, 'missing_values': True, 'type': 'String', 'field': x} for x in ref_cols]


real_match_cols = [pair['ref'] for pair in match_cols]

for col in real_match_cols:
    source[col] = pd_pre_process(source[col], remove_punctuation=True)
    ref[col] = pd_pre_process(ref[col], remove_punctuation=True)

# Replace np.nan 's by None 's
source = source.where(source.notnull(), None)
ref = ref.where(ref.notnull(), None)

source_items = source[ref_cols].to_dict('index')
ref_items = ref[ref_cols].to_dict('index')

datamodel = dedupe.datamodel.DataModel(fields)
my_predicates = list(datamodel.predicates(index_predicates=False, canopies=False)) # TODO: set to True

compound_length = 1

blocker = blocking.Blocker(my_predicates)

def invert_predicate_cover(dupe_cover):
    inv_dupe_cover = defaultdict(set)
    for key, matches in dupe_cover.items():
        for match_id in matches:
            inv_dupe_cover[match_id].add(key)
    return inv_dupe_cover


def my_print(candidate):
    keys = candidate[0].keys()
    print('*****\n')
    for key in keys:
        print(candidate[0][key])
        print(candidate[1][key])

def my_other_print(record):
    for key, value in record.items():
        print(value)


def pre_process_string(string):
    TO_REPLACE = [('lycee', ''), (' de ', ' ')]
    string = unidecode.unidecode(string.lower())
    for pair in TO_REPLACE:
        string = string.replace(pair[0], pair[1])
    return string


def make_n_cover(dupe_cover, n):
    dupe_cover_n = dict() 
    for x in itertools.combinations(dupe_cover.keys(), n):
        elems = dupe_cover[x[0]]
        for i in range(1, n):
            elems = elems & dupe_cover[x[i]]
        if elems:
            dupe_cover_n[x] = elems
    return dupe_cover_n






#def score(id_source, id_ref):
#    pair = {'source': source_items[id_source], 'ref': ref_items[id_ref]}
#    # score = sum(crfEd(pair['source'][col], pair['ref'][col]) for col in real_match_cols) / len_match_cols
#    crfEd = CRFEditDistance()
#    score = crfEd(pair['source']['full_name'], pair['ref']['full_name'])
#    return score



#def get_best_pair(source_items, ref_items, source_id, ref_ids):
#    '''Returns the source_id, ref_id pairs with the smallest edit distance'''
#    if len(ref_ids) > 50:
#        raise RuntimeWarning('ref_ids is too large (> 50)')
#        
#    if len(ref_ids) == 1:
#        ref_id = ref_ids[0]
#    else:
#        scores = [(ref_id, score(source_id, ref_id)) for ref_id in ref_ids]
#        ref_id = min(scores, key=lambda x: x[1])[0]
#    return (source_id, ref_id)


    

'''
loop:
1) Choose predicate (50% with highest ratio ; 50% random not)
2) Choose a sample (include string distance ?)
3) Update predicate_info including precision, recall(exclude current) and
    ratio (and sort ?)

WARNING: elements are poped from predicate_cover    



*****  

We want to learn:

1) Good blocking rule
2) Good string distance (consider it to be a classifier) for this blocking rule.

Ideas:
Below threshold doesn't influence precision
Keep string distance in labels and simultaneously learn the string distance
In info keep all string distances and compute threshold from 2 matches
and 2 non matches up.
Compute threshold on best block each time    

1) Build string distance CLASSIFIER on ALL examples
2) Blocker rules: Compute Precision using new info. RECALL doesn't change. Compute RATIO
3) 50% Choose best predicate (highest RATIO) and best sample (highest probability with distance CLASSIFIER) 

Classifier should be corrected for precision of blocking
Ratio should include absolute precision of blocking (without string distance)

'''

from sklearn import linear_model

class Labeller():
    def __init__(self, candidates, blocker, datamodel, n=3):
        self.distances = datamodel.distances(candidates)
        self.classifier = self._init_classifier(self)
        
        self.candidates = candidates
        self.labelled = []
        self.predicate_cover = cover(blocker, candidates, 1)
        self.predicate_cover = make_n_cover(self.predicate_cover, n)
        self.predicate_info = {key: {'key': key, 
                                     'num_labelled': 0, # Number of times a selected pair had this predicate
                                     'num_labelled_selected': 0, # Number of times selected as predicate
                                     'num_matches': 0, # Number of matches found
                                     'num_matches_selected': 0, # Number of matches found when predicate was selected
                                     'has_pairs': True, # Still has pairs to label
                                     'ratio': 0.001} \
                                      for key in self.predicate_cover.keys()}
        
        self.candidate_cover = invert_predicate_cover(self.predicate_cover)
        self.num_matches = 0
        self.num_labelled = 0
    
    
    def _init_classifier(self):
        classifier = linear_model.LogisticRegression()
        classifier.intercept_ = np.array([0])
        classifier. coef_ = np.array([[0.5 for _ in range(self.distances.shape[1])]])
        return classifier
    
    def train_classifier(self):      
        X = self.distances[[x['pair_id'] for x in self.labelled]]
        Y = [x['match'] for x in self.labelled]
        self.classifier.fit(X, Y)
        
        
    def update(self, selected_predicate, pair_id, is_match):
        # Update global count
        self.num_labelled += 1
        self.num_matches += int(is_match)
        
        # Add count of selected
        self.predicate_info[selected_predicate]['num_labelled_selected'] += 1
        self.predicate_info[selected_predicate]['num_matches_selected'] += int(is_match)
        
        for key in self.predicate_info.keys():
            # If the selected pair is covered by the predicate
            if key in self.candidate_cover[pair_id]:
                # One more label for this predicate
                self.predicate_info[key]['num_labelled'] += 1
                
                # Add match for this predicate (if is_match)
                if is_match:
                    self.predicate_info[key]['num_matches'] += 1
                
                # Update precision
                self.predicate_info[key]['precision'] = self.predicate_info[key]['num_matches'] \
                                                 / self.predicate_info[key]['num_labelled']

                # Remove the pair from predicate_cover          
                self.predicate_cover[key] = self.predicate_cover[key] - {pair_id} 
                
                # Check if it still has pairs
                self.predicate_info[key]['has_pairs'] = bool(len(self.predicate_cover[key]))
            
            # Update recall (on examples not selected using this predicate)
            if (self.num_matches - self.predicate_info[key]['num_matches_selected']):
                self.predicate_info[key]['recall'] = \
                    (self.predicate_info[key]['num_matches'] - self.predicate_info[key]['num_matches_selected']) / (self.num_matches - self.predicate_info[key]['num_matches_selected'])
                    
            # Compute ratio
            self.predicate_info[key]['ratio'] = self.predicate_info[key].get('precision', 0.1) \
                                        * self.predicate_info[key].get('recall', 0.1)  
                               


    def get_best_predicate(self):
        '''Returns the predicate that has the highest ratio'''
        return max(self.predicate_info.values(), key=lambda x: x['ratio'] * x['has_pairs'])['key']

    def choose_pair(self, proba=0.5):
        rand_val = random.random()
        if rand_val <= proba:
            # Choose best_predicate
            print('Getting best')
            predicate = self.get_best_predicate()
        else:
            for _ in range(100):
                predicate = random.choice(list(self.predicate_info.keys()))
                if self.predicate_info[predicate]['has_pairs']:
                    break
            else:
                raise RuntimeError('Could not found predicate with pairs')
            
        pair_id = self.predicate_cover[predicate].pop()        
        return predicate, pair_id

# Load prelabelled data for testing
with open('temp_labelling.json') as f:
    real_labelled = json.load(f)

# Candidates of pairs to be labelled
candidates = [pair['candidate'] for pair in real_labelled]


# Initiate
labeller = Labeller(candidates, blocker)

quit_ = False
while not quit_:
    selected_predicate, pair_id = labeller.choose_pair(0.5)
    
    pprint.pprint(labeller.predicate_info[selected_predicate])
    my_print(labeller.candidates[pair_id])
    
    while True:
        input_ = input('>')
        if input_ == 'y':
            is_match = True
            break
        elif input_ == 'n':
            is_match = False
            break
        elif input_ == 'f':
            quit_ = True
            break
        else:
            print('(y)es, (n)o, or (f)inished')
            
    if input_ in ['y', 'n']:
        labeller.update(selected_predicate, pair_id, is_match)
        #labelled.append({'candidate': candidates[pair_id], 'match': is_match})

assert False

tab = df_from_predicate_cover(predicate_cover)
tab['col'] = tab.predicate.apply(get_col)
tab.groupby('col')['count'].mean() # meaningfull columns

# Create double key
double_predicate_cover = make_n_cover(predicate_cover, 2)
double_tab = df_from_predicate_cover(double_predicate_cover)

# Create triple key
triple_predicate_cover = make_n_cover(predicate_cover, 3)
triple_tab = df_from_predicate_cover(triple_predicate_coverr)