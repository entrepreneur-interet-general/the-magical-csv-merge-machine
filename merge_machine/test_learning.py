#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:18:51 2017

@author: leo
"""

from collections import defaultdict, deque
import functools
import itertools
import os
from string import punctuation
import random
import dedupe
from dedupe import sampling
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

def blockedSample(sampler, sample_size, predicates, *args) :
    
    blocked_sample = set()
    remaining_sample = sample_size - len(blocked_sample)
    previous_sample_size = 0

    while remaining_sample and predicates :
        random.shuffle(predicates)

        new_sample = sampler(remaining_sample, 
                             predicates,
                             *args)

        filtered_sample = (subsample for subsample 
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

            
        predicates = [pred for pred, pred_sample 
                      in zip(predicates, new_sample)
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

def subsample(total_size, predicates) :
    splits = evenSplits(total_size, len(predicates))
    for split, predicate in zip(splits, predicates) :
        yield split, predicate

def linkSamplePredicates(sample_size, predicates, items1, items2) :
    n_1 = len(items1)
    n_2 = len(items2)

    for subsample_size, predicate in subsample(sample_size, predicates) :
        
        if not subsample_size :
            yield None
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

        yield linkSamplePredicate(subsample_size, predicate, items1, items2)

linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) 


dir_path = '/home/leo/Documents/eig/the-magical-csv-merge-machine/merge_machine/local_test_data'

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
    source[match['source']] = pd_pre_process(source[match['ref']], remove_punctuation=True)
    ref[match['ref']] = pd_pre_process(ref[match['ref']], remove_punctuation=True)

# Replace np.nan 's by None 's
source = source.where(source.notnull(), None)
ref = ref.where(ref.notnull(), None)

source_items = source[ref_cols].to_dict('index')
ref_items = ref[ref_cols].to_dict('index')

deque_1 = sampling.randomDeque(source_items)
deque_2 = sampling.randomDeque(ref_items)

datamodel = dedupe.datamodel.DataModel(fields)
predicates = list(datamodel.predicates(False, False))

blocked_sample_keys = linkBlockedSample(5000,
                                         predicates,
                                         deque_1,
                                         deque_2)








