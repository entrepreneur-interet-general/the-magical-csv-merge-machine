#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:18:51 2017

@author: leo

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

1) Build string distance CLASSIFIER on ALL examples and make prediction for each example
2) Blocker rules: Compute Precision using new info. RECALL doesn't change. Compute RATIO
3) 50% Choose best predicate (highest RATIO) and best sample (highest probability with distance CLASSIFIER) 

Classifier should be corrected for precision of blocking
Ratio should include absolute precision of blocking (without string distance)

Choose pairs based on classifier probability
"""

from future.utils import viewitems

from collections import defaultdict, deque
import functools
import itertools
import json
import math
import os
from string import punctuation
import random

import dedupe
from dedupe import blocking, predicates, sampling
import numpy as np
import pandas as pd
import pprint
from sklearn import linear_model
import unidecode



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

# =============================================================================
# ^ Up is from dedupe
# =============================================================================

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

def prepare_data(source, ref, match_cols):    
    ref_cols = [x['ref'] for x in match_cols]
    
    temp_match_cols = {x['source']: x['ref'] for x in match_cols}
    
    # Replace column_names in source by those in ref
    source.columns = [temp_match_cols.get(x, x) for x in source.columns]
    
    real_match_cols = [pair['ref'] for pair in match_cols]
    
    for col in real_match_cols:
        source[col] = pd_pre_process(source[col], remove_punctuation=True)
        ref[col] = pd_pre_process(ref[col], remove_punctuation=True)
    
    # Replace np.nan 's by None 's
    source = source.where(source.notnull(), None)
    ref = ref.where(ref.notnull(), None)
    
    source_items = source[ref_cols].to_dict('index')
    ref_items = ref[ref_cols].to_dict('index')
    return source_items, ref_items

def gen_dedupe_datamodel(ref_cols):
    fields = [{'crf': True, 'missing_values': True, 'type': 'String', 'field': x} for x in ref_cols]
    dedupe_datamodel = dedupe.datamodel.DataModel(fields)
    return dedupe_datamodel


source_path = os.path.join('local_test_data', 'source.csv')
ref_path = os.path.join('local_test_data', 'ref.csv')

match_cols = [{'source': 'departement', 'ref': 'departement'},
              {'source': 'commune', 'ref': 'localite_acheminement_uai'},
              {'source': 'lycees_sources', 'ref': 'full_name'}]

real_match_cols = [pair['ref'] for pair in match_cols]

source = pd.read_csv(source_path, dtype=str)
ref = pd.read_csv(ref_path, dtype=str)
source_items, ref_items = prepare_data(source, ref, match_cols)


dedupe_datamodel = gen_dedupe_datamodel(real_match_cols)
my_predicates = list(dedupe_datamodel.predicates(index_predicates=False, canopies=False)) # TODO: set to True

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

def make_n_cover(dupe_cover, n):
    '''Take a single dupe_cover and creates cover for intersection of covers'''
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



class IDFDistance():
    ''''Class to compute distances based on inverse document frequency'''
    def __init__(self, ref, columns):
        self.columns = columns
        self.idfs = dict()
        for col in self.columns:
            self.idfs[col] = self._gen_idf(ref, col)
        
        self.default_value = 0
        self.default_idf_source = 8.7
        self.default_idf_ref = 8.7
    
    @staticmethod        
    def _gen_idf(ref, col):
        '''Generate IDF dict (key is word, value is idf)'''
        word_counts = defaultdict(int)
        num_docs = ref.shape[0]

        all_words = ref[col].str.cat(sep=' ').split()
        for word in all_words:
            word_counts[word] += 1
        
        idf = dict()
        for word, word_count in word_counts.items():
            idf[word] =  math.log((num_docs / word_count))
        return idf
    
    @staticmethod
    def _tokenize(string):
        return set(string.split())
    
    def idf_cosine(self, idf, string_1, string_2):
        '''Computes cosine distance between string_1 and string_2'''
        if (string_1 is None) or (string_2 is None):
            return self.default_value
        
        tokens_1 = self._tokenize(string_1)
        tokens_2 = self._tokenize(string_2)
        common = tokens_1 & tokens_2
        
        val_0 = sum(idf[word]**2 for word in common)
        val_1 = math.sqrt(sum(idf.get(word, self.default_idf_source)**2 for word in tokens_1))
        val_2 = math.sqrt(sum(idf.get(word, self.default_idf_ref)**2 for word in tokens_2))
        
        return 1 - val_0 / (val_1 * val_2)
        
    def distances(self, candidates):
        '''
        Computes matrix of distances for all fields for all records in candidates
        '''
        distances = []
        for candidate in candidates:
            pair_distances = []
            for col in self.columns:
                string_1 = candidate[0][col]
                string_2 = candidate[1][col]
                pair_distances.append(self.idf_cosine(self.idfs[col], string_1, string_2))
            distances.append(pair_distances)
        return np.array(distances)


class Learner():
    '''
    Class to learn generate training labels while learning good blocking
    predicates and classifier.    
    '''
    def __init__(self, candidates, blocker, datamodel, n=3, min_pairs_for_classifier=10**9):
        self.distances = datamodel.distances(candidates)
        self.classifier = self._init_classifier()
        self.min_pairs_for_classifier = min_pairs_for_classifier
        self.candidates = candidates
        self.is_match_from_classifier = np.array([True]*len(candidates))
        self.is_match = dict()
        
        
        # Include equivalent predicates
        all_predicate_cover = cover(blocker, candidates, 1)
        all_predicate_cover = make_n_cover(all_predicate_cover, n) # TODO: hash before to save time ?
        
        self.predicate_cover = self._restrict_predicate_cover(all_predicate_cover)


        
        self.predicate_info = {key: {'key': key, 
                                     'labelled_pairs': set(), # Labelled pairs covered by this predicate
                                     'labelled_pairs_selected': set(), # Pairs selected precisely to test this preidcate
                                     'has_pairs': True, # Still has pairs to label
                                     'recall': 0.1,
                                     'precision': 0.1,
                                     'ratio': 0.001} \
                                      for key in self.predicate_cover.keys()}
        
        self.candidate_cover = invert_predicate_cover(self.predicate_cover)
        self.num_matches = 0
        self.num_labelled = 0
        
        self.best_predicate = list(self.predicate_cover.keys())[0]
    
    @staticmethod
    def _restrict_predicate_cover(all_predicate_cover):
        '''
        Returns a version of predicate cover containing only predicates with
        distinct covers (based on set string representation hash)
        '''
        hashes_included = set()
        predicate_cover = dict()
        for predicate, cover_ in all_predicate_cover.items():
            hash_ = hash(cover_.__str__())
            if hash_ not in hashes_included:
                predicate_cover[predicate] = cover_
                hashes_included.add(hash_)        
        return predicate_cover
    
    def _init_classifier(self):
        '''Initiate the distance-based classifier'''
        #        classifier = linear_model.SGDClassifier(class_weight="balanced", 
        #                                                     fit_intercept=True, 
        #                                                     loss='modified_huber')
        classifier = linear_model.LogisticRegressionCV(class_weight="balanced", 
                                                     fit_intercept=True)
        classifier.intercept_ = np.array([0.])
        classifier.coef_ = np.array([[0.5 for _ in range(self.distances.shape[1])]])
        return classifier
    
    def train_classifier(self):
        '''Train the distance-based classifier on all labelled pairs'''
        pair_match = list(self.is_match.items())
        X = self.distances[[pair_id for (pair_id, _) in pair_match]]
        Y = [is_match for (_, is_match) in pair_match]
        self.classifier.fit(X, Y)
        
    def update(self, selected_predicate, pair_id, is_match):
        '''
        Update both global (counts and classifier) and predicate specific 
        information using the new label provided for the pair.
        
        INPUT: 
            - selected_predicate: predicate used to generate pair_id
            - pair_id: id of the pair that was labelled
            - is_match: whether or not the pair is a match or not
        '''
        
        # Update global count
        self.num_labelled += 1
        self.num_matches += int(is_match)
        self.is_match[pair_id] = is_match
        
        # Add count of selected
        self.predicate_info[selected_predicate]['labelled_pairs_selected'].add(pair_id)
        
        # Update matches_from classifier
        use_classifier = (self.num_labelled - self.num_matches >= self.min_pairs_for_classifier) and self.num_matches
        if use_classifier:
            self.train_classifier()
            self.is_match_from_classifier = self.classifier.predict(self.distances)

        # Update_blocking rules
        for predicate, this_predicate_info in self.predicate_info.items():
            if predicate in self.candidate_cover[pair_id]:
                this_predicate_info['labelled_pairs'].add(pair_id)
        
                # Remove the pair from predicate_cover          
                self.predicate_cover[predicate] = self.predicate_cover[predicate] - {pair_id} 
                
                # Check if it still has pairs
                this_predicate_info['has_pairs'] = bool(len(self.predicate_cover[predicate]))        
        
        
            # Compute pseudo-precision (classifier + blocking precision)
            num_covered_matches = sum(self.is_match_from_classifier[pair_id] \
                                  and self.is_match[pair_id] \
                                  for pair_id in this_predicate_info['labelled_pairs'])
            num_covered = sum(self.is_match_from_classifier[pair_id] \
                                  for pair_id in this_predicate_info['labelled_pairs'])

        
            if num_covered:
                this_predicate_info['precision'] = num_covered_matches / num_covered
            else:
                this_predicate_info['precision'] = 0.1
            
            # Compute pseudo-recall (blocking only, exlcuding matches from selected predicate)
            num_covered_matches = sum(self.is_match[pair_id] \
                                  for pair_id in this_predicate_info['labelled_pairs'])      
            num_covered_matches_selected = sum(self.is_match[pair_id] \
                                  for pair_id in this_predicate_info['labelled_pairs_selected'])    
            if (self.num_matches-num_covered_matches_selected) >= 2:
                this_predicate_info['recall'] = (num_covered_matches-num_covered_matches_selected) \
                                                / (self.num_matches-num_covered_matches_selected)   
            
            # Compute ratio$
            this_predicate_info['ratio'] = this_predicate_info['precision'] * this_predicate_info['recall']

    def get_sorted_predicates(self):
        '''Returns the predicate that has the highest ratio'''
        return [x['key'] for x in sorted(self.predicate_info.values(), \
                key=lambda x: x['ratio'] * x['has_pairs'], reverse=True)]

    def choose_pair(self, proba=0.5):
        '''
        Returns a pair to label
        
        INPUT:
            - proba: probability with which the pair is selected from cover of 
                    the current best predicate
        
        '''
        rand_val = random.random()
        
        sorted_predicates = self.get_sorted_predicates()
        self.best_predicate = sorted_predicates[0]
        if rand_val <= proba:
            # Choose best_predicate
            print('Getting best')
            predicate = self.best_predicate 
            pair_ids = self.predicate_cover[predicate]
        else:
            for predicate in sorted_predicates[1:]:
                pair_ids = self.predicate_cover[predicate] - self.predicate_cover[self.best_predicate]
                if pair_ids:
                    break
            else:
                raise RuntimeError('Could not found predicate with pairs')      
    
        pair_id = random.choice(list(pair_ids))
        return predicate, pair_id
    

    def compute_ratio(self, precision, recall, target_precision=0.85):
        if (precision >= target_precision) or (self.num_labelled <= 10):
            return precision * recall
        else:
            return target_precision * (precision/target_precision)**3 * recall

    def score_predicate(self, predicate, real_labelled):
        '''
        If real labels are available, scores the given predicate (precision, 
        recall, ratio)        
        '''
        # Evaluate predicate
        guessed_as_dupes = {x for x in self.predicate_cover[predicate] \
                            if self.is_match_from_classifier[x]}
        
        actual_dupes = {i for i, x in enumerate(real_labelled) if x['match']}
        
        precision = len(guessed_as_dupes & actual_dupes) / max(len(guessed_as_dupes), 1)
        recall = len(guessed_as_dupes & actual_dupes) / max(len(actual_dupes), 1)
        ratio = self.compute_ratio(precision, recall)
        
        return precision, recall, ratio



class CompoundDatamodel():
    '''
    Creates an object with a "distances" method that concatenates results
    of the "distances" methods of all objects passed as arguments for __init__
    '''
    def __init__(self, *datamodels):
        self.datamodels = datamodels
        
    def distances(self, candidates):
        return np.concatenate(list(dm.distances(candidates) for dm in self.datamodels), 1)


def label(learner, real_labelled=None, verbose=True):
    '''Human interface for labeling data'''
    manual_labelling = real_labelled is None
    
    quit_ = False
    hist_metrics = []
    final_metrics = []
    while (not quit_) \
            and learner.num_labelled <= max_labels\
            and learner.num_matches <= num_matches:
        predicate, pair_id = learner.choose_pair(prop)

        if verbose:
            pprint.pprint({key: val for key, val in learner.predicate_info[predicate].items() if 'labelled_pairs' not in key})
            print('Num matches: {0} ; Num labelled: {1}'.format(learner.num_matches, learner.num_labelled))
            print('Is match from classifier:', learner.is_match_from_classifier[pair_id])
            print('Is match from real_labelling', real_labelled[pair_id]['match'])    
            my_print(learner.candidates[pair_id])
            
        hist_metrics.append(copy.deepcopy(learner.predicate_info[learner.best_predicate]))
        final_metrics.append(learner.predicate_info[learner.best_predicate])
        
        if manual_labelling:
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
                learner.update(predicate, pair_id, is_match)
        else:
            learner.update(predicate, pair_id, real_labelled[pair_id]['match'])
            #labelled.append({'candidate': candidates[pair_id], 'match': is_match})
    return learner


import copy

manual_labelling = False
max_labels = 100
prop = 0.65
n = 3
min_pairs_for_classifier = 3

num_matches = 12

# Load prelabelled data for testing
with open('temp_labelling.json') as f:
    real_labelled = json.load(f)
random.shuffle(real_labelled)    

# Candidates of pairs to be labelled
candidates = [pair['candidate'] for pair in real_labelled]



# Initiate
my_datamodel = IDFDistance(ref, real_match_cols)
datamodel = CompoundDatamodel(dedupe_datamodel, my_datamodel)
learner = Learner(candidates, blocker, datamodel, n, min_pairs_for_classifier)

# Perform latbelling
learner = label(learner, real_labelled)

print('\n')

# Print realest metric
final_predicate = learner.best_predicate
precision, recall, ratio = learner.score_predicate(final_predicate, real_labelled)
print(final_predicate)
print('Precision: {0} ; Recall: {1} ; Ratio: {2}'.format(precision, recall, ratio))



classifier = learner.classifier
predicate = predicates.CompoundPredicate(learner.best_predicate)

# Hash to set of ref_id's
hash_to_ref = defaultdict(set)
for ref_id, record in ref_items.items():
    for hash_ in predicate(record):
        hash_to_ref[hash_].add(ref_id)

# Source id's to set of ref_id's
source_candidates = defaultdict(set)
for source_id, record in source_items.items():
    for hash_ in predicate(record):
        source_candidates[source_id] = source_candidates[source_id].union(hash_to_ref[hash_])

# Look at block sizes
block_sizes = pd.Series([len(value) for key, value in source_candidates.items()]).describe([x/10. for x in range(10)])
print(block_sizes)


def get_distances(datamodel, classifier, id_source, candidates):
    source_record = source_items[id_source]
    ref_records = [ref_items[id_ref] for id_ref in candidates]
    distances = datamodel.distances([(source_record, ref_record) for ref_record in ref_records])
    return distances

source_best_match = dict()
for id_source, candidates in source_candidates.items():
    if len(candidates) == 0:
        source_best_match[id_source] = None
    elif len(candidates) == 1:
         id_ref = next(iter(candidates))
         source_best_match[id_source] = id_ref
    else:
        list_candidates = list(candidates)
        distances = get_distances(datamodel, classifier, id_source, list_candidates)
        scores = zip(list_candidates, classifier.predict_proba(distances)[:,1])
        best_score = max(scores, key=lambda x: x[1])
        if best_score[1] >= 0.5:
            source_best_match[id_source]  = best_score[0]
#        else:
#            source_best_match[id_source] = None


# Print all matches found
errors_only = True
for id_source, id_ref in source_best_match.items():
    if id_ref is not None:
        
        is_match = source['uai'].iloc[id_source] == ref['numero_uai'].iloc[id_ref]
        if (not is_match) or (not errors_only):
            my_print((source_items[id_source], ref_items[id_ref]))
            print('  --> src: {0} ; ref: {1} ; is_match: {2}'.format(id_source, id_ref, is_match))

def anal_print(id_source):
    print('\n' + '>'*20 + ' id_source: {0} \nMATCH FOUND'.format(id_source))
    id_ref_good = source_best_match[id_source]
    my_print((source_items[id_source], ref_items[id_ref_good]))
    print('  -> id_ref: {0} ; score: {1}'.format(id_ref_good, score(id_source, id_ref_good)))
    
    print('\n>>>>>>>>\nOTHER OPTIONS IN BLOCKS')
    for id_ref in source_candidates[id_source]:
        if id_ref != id_ref_good:
            my_other_print(ref_items[id_ref])
            print('  -> id_ref: {0} ; score: {1}'.format(id_ref, score(id_source, id_ref)))

def _get_ref_uai(id_source):
    if source_best_match.get(id_source, None) is not None:
        return ref.loc[source_best_match[id_source], 'numero_uai']
    else:
        return np.nan

source['ref_uai'] = [_get_ref_uai(id_source) for id_source in source.index]
source['good'] = source['uai'].str.upper() == source['ref_uai'].str.upper()


precision = source.loc[source.ref_uai.notnull(), 'good'].mean()
recall = source.ref_uai.notnull().mean()
print(final_predicate)
print('Precision: {0} ; Recall: {1}'.format(precision, recall))

#for id_source in source[~source.good & source.ref_uai.notnull()].index:
#    anal_print(id_source)

# Print errors



assert False

# Explore bad result
source_id = 1399
print('\nREF')
for ref_id in source_candidates[source_id]:
    print(ref_items[ref_id], '\n')
    print(predicates.CompoundPredicate(learner.best_predicate)(ref_items[ref_id]))
    score = learner.classifier.predict_proba(get_distances(datamodel, classifier, source_id, [ref_id]))[0, 1]
    print(' --> score:', score, '\n')
print('SOURCE:\n', source_items[source_id], '\n')
print(predicates.CompoundPredicate(learner.best_predicate)(source_items[source_id]))
print('\n', learner.best_predicate)   

# Should have been
ref_id = 1581
print('\n', ref_items[ref_id])
print(predicates.CompoundPredicate(learner.best_predicate)(ref_items[ref_id]))

assert False

'''
Ideas:
    
- do not store most infrequent and use get install
- index with 1 insertion distance
'''

import math

col = 'full_name'



assert False

# Print best_predicate evolution
next_id = 0
class_id = dict()
best_ids = []
for predicate_info in hist_metrics:
    predicate = predicate_info['key']
    if predicate not in class_id:   
        class_id[predicate] = next_id
        next_id += 1
    best_ids.append(class_id[predicate])
print(best_ids)


# Print final estimated metrics
for metric in ['precision', 'recall', 'ratio']:
    print('\nMetric:', metric)
    print([predicate_info[metric] for predicate_info in hist_metrics])
    print('\n')
    print([predicate_info[metric] for predicate_info in final_metrics])


# Evaluate real predicate metrics as function of iterations
predicate_metrics = dict()
for x in hist_metrics:
    predicate = x['key']
    if predicate not in predicate_metrics:
        predicate_metrics[predicate] = learner.score_predicate(predicate, real_labelled)

real_metric = [predicate_metrics[x['key']] for x in hist_metrics]


# Print real score for top predicates 
sorted_predicates = learner.get_sorted_predicates()
for predicate in sorted_predicates[:10]:
    print(learner.score_predicate(predicate, real_labelled))
    
# Check influence of using multiple dupes
actual_dupes = {i for i, x in enumerate(real_labelled) if x['match']}

for predicate in sorted_predicates[1:]:
    pair_ids = learner.predicate_cover[predicate] - learner.predicate_cover[learner.best_predicate]
    precision = len(pair_ids & actual_dupes) / max(len(pair_ids), 1)    
    if (precision >= 0.8) and (len(pair_ids) >= 1):
        print({key: value for key, value in learner.predicate_info[predicate].items() if 'labelled' not in key})
        print('Extra pairs:', len(pair_ids), ' | precision: ', precision)
    
    
    
# src: 1368 ; ref: 1513
    