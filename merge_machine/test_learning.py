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
import os
from string import punctuation
import random
import dedupe
from dedupe import blocking, predicates, sampling
import numpy as np
import pandas as pd
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



for match in match_cols:
    source[match['ref']] = pd_pre_process(source[match['ref']], remove_punctuation=True)
    ref[match['ref']] = pd_pre_process(ref[match['ref']], remove_punctuation=True)

# Replace np.nan 's by None 's
source = source.where(source.notnull(), None)
ref = ref.where(ref.notnull(), None)

source_items = source[ref_cols].to_dict('index')
ref_items = ref[ref_cols].to_dict('index')

deque_1 = sampling.randomDeque(source_items)
deque_2 = sampling.randomDeque(ref_items)

datamodel = dedupe.datamodel.DataModel(fields)
my_predicates = list(datamodel.predicates(index_predicates=False, canopies=False)) # TODO: set to True

blocked_sample_keys = linkBlockedSample(5000,
                                         my_predicates,
                                         deque_1,
                                         deque_2)



#candidates = [(source[k1], ref[k2])
#               for k1, k2
#               in blocked_sample_keys | random_sample_keys]

candidates = [(source_items[k1], ref_items[k2])
               for predicate, (k1, k2)
               in blocked_sample_keys]


compound_length = 1

blocker = blocking.Blocker(my_predicates)

blocker.indexAll({i : record
                       for i, record
                       in enumerate(unroll(candidates))})
    
dupe_cover = cover(blocker, candidates, compound_length)

dupe_cover_count = {key: len(predicates) for key, predicates in dupe_cover.items()}

tab = pd.DataFrame([[key, val] for key, val in dupe_cover_count.items()], columns=['predicate', 'count'])
tab['col'] = tab.predicate.apply(get_col)
tab.groupby('col')['count'].mean() # meaningfull columns

def invert_dupe_cover(dupe_cover):
    inv_dupe_cover = defaultdict(set)
    for key, matches in dupe_cover.items():
        for match_id in matches:
            inv_dupe_cover[match_id].add(key)
    return inv_dupe_cover

inv_dupe_cover = invert_dupe_cover(dupe_cover)

inv_dupe_cover_count = {key: len(predicates) for key, predicates in inv_dupe_cover.items()}
set(my_predicates) - set(dupe_cover.keys())

import re

word_count = pd.Series(re.findall(r"[\w']+", source.full_name.str.lower().str.cat(sep=' '))).value_counts()


def my_print(candidate):
    keys = candidate[0].keys()
    print('*****\n')
    for key in keys:
        print(candidate[0][key])
        print(candidate[1][key])

def my_other_print(record):
    print('*****\n')
    for key, value in record.items():
        print(value)

def excl_1(candidate):
    '''Return true if candidate should be excluded'''
    dep_1 = candidate[0]['departement']
    dep_2 = candidate[1]['departement']
    return (len(dep_1) == 2) \
            and (len(dep_2) == 3) \
            and (dep_2[0] == '0') \
            and (dep_1 != dep_2[1:])

def n_grams(string, N):
    return {string[i:i+N] for i in range(len(string)-N+1)}


def pre_process_string(string):
    TO_REPLACE = [('lycee', ''), (' de ', ' ')]
    string = unidecode.unidecode(string.lower())
    for pair in TO_REPLACE:
        string = string.replace(pair[0], pair[1])
    return string

def excl_2(candidate, field):
    field_1 = candidate[0][field]
    field_2 = candidate[1][field]
    
    ngrams_1 = n_grams(pre_process_string(field_1), 3)
    ngrams_2 = n_grams(pre_process_string(field_2), 3)
    
    return not (ngrams_1 & ngrams_2)
    

labelled = []
cursor = 0


for candidate in candidates[cursor:]:
    if (None in candidate[0].values()) or (None in candidate[1].values()):
        continue
    
    my_print(candidate)
    
    if excl_1(candidate) \
        or excl_2(candidate, 'full_name') \
        or excl_2(candidate, 'localite_acheminement_uai'):
        labelled.append({'candidate': candidate, 'match': False})
    else:
        is_match = None
        while is_match not in ['0', '1', 'stop']:
            if is_match is not None:
                print('stop to stop')
            is_match = input('\n({0}) >'.format(cursor))
        if is_match == 'stop':
            print('Done labelling. Did {0}'.format(cursor))
            break
        labelled.append({'candidate': candidate, 'match': is_match=='1'})
    cursor += 1
    


if False:
    import json
    with open('temp_labelling.json', 'w') as w:
        json.dump(labelled, w)



def count_matches(predicates, labelled):
    return sum(labelled[x]['match'] for x in predicates)

def df_from_dupe_cover(dupe_cover):
    dupe_cover_match_count = {key: count_matches(predicates, labelled) \
                              for key, predicates in dupe_cover.items()}
    dupe_cover_count = {key: len(predicates) for key, predicates in dupe_cover.items()}
    
    tab = pd.DataFrame([[key, val, dupe_cover_match_count[key]] \
                        for key, val in dupe_cover_count.items()], \
                        columns=['predicate', 'count', 'match_count'])
    
    tab['precision'] = tab['match_count'] / tab['count']
    match_count = sum(x['match'] for x in labelled)
    tab['recall'] = tab.match_count / match_count
    tab.sort_values('recall', ascending=False, inplace=True)
    
    tab['ratio'] = tab['recall'] * tab['precision']
    
    return tab


def make_n_cover(dupe_cover, n):
    dupe_cover_n = dict() 
    for x in itertools.combinations(dupe_cover_count, n):
        elems = dupe_cover[x[0]]
        for i in range(1, n):
            elems = elems & dupe_cover[x[i]]
        if elems:
            dupe_cover_n[x] = elems

    return dupe_cover_n


candidates = [x['candidate'] for x in labelled]
dupe_cover = cover(blocker, candidates, compound_length)


tab = df_from_dupe_cover(dupe_cover)
tab['col'] = tab.predicate.apply(get_col)
tab.groupby('col')['count'].mean() # meaningfull columns

# Create double key
double_dupe_cover = make_n_cover(dupe_cover, 2)
double_tab = df_from_dupe_cover(double_dupe_cover)

# Create triple key
triple_dupe_cover = make_n_cover(dupe_cover, 3)
triple_tab = df_from_dupe_cover(triple_dupe_cover)


predicate = triple_tab['predicate'].iloc[-1]
for id_ in triple_dupe_cover[predicate]:
    if not labelled[id_]['match']:
        my_print(labelled[id_]['candidate'])
        
        
        
compound_predicate = predicates.CompoundPredicate(predicate)


num_indexes = pd.Series(len(compound_predicate(x[0])) for x in candidates)

my_index = defaultdict(set)
for i, label in enumerate(labelled):
    for pos in [0, 1]:
        for index in compound_predicate(label['candidate'][pos]):
            my_index[index].add((i, pos, label['match']))
            
my_index_small = {key: value for key, value in my_index.items() if len(value)>1}    



id_ = 'hirson:oliotcurie:2'
for (i, pos, _) in my_index_small[id_]:
    my_other_print(labelled[i]['candidate'][pos])