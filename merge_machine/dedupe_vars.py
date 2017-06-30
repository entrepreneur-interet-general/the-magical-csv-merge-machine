#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:21:11 2017

@author: m75380


# Learning
deduper.uncertainPairs -->  deduper.active_learner.get() looks at probabilities

markPairs: basically calls mark (fit on result of self.transform)

transform --> self.data_model.distances(pairs)
distances --> 

# 

Sample() class: just random sampling if overflow

deque (collections): double ending queue (like list)
randomDeque: creates deque with random sample of data


linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) # linkSamplePredicates replaces first argument of blockedSample

predicates = [SimplePredicate: (fingerprint, departement), 
              SimplePredicate: (sortedAcronym, departement), 
              SimplePredicate: (fingerprint, localite_acheminement_uai), 
              SimplePredicate: (tokenFieldPredicate, localite_acheminement_uai), 
              SimplePredicate: (commonSixGram, departement), 
              SimplePredicate: (commonThreeTokens, departement), 
              SimplePredicate: (metaphoneToken, departement), 
              SimplePredicate: (commonIntegerPredicate, localite_acheminement_uai), 
              SimplePredicate: (metaphoneToken, localite_acheminement_uai), 
              SimplePredicate: (commonIntegerPredicate, full_name), 
              SimplePredicate: (oneGramFingerprint, departement), 
              SimplePredicate: (suffixArray, full_name), 
              SimplePredicate: (sameFiveCharStartPredicate, localite_acheminement_uai), 
              SimplePredicate: (sameFiveCharStartPredicate, departement), 
              SimplePredicate: (tokenFieldPredicate, departement), 
              SimplePredicate: (commonThreeTokens, localite_acheminement_uai), 
              SimplePredicate: (oneGramFingerprint, full_name), 
              SimplePredicate: (nearIntegersPredicate, localite_acheminement_uai), 
              SimplePredicate: (commonIntegerPredicate, departement), 
              SimplePredicate: (wholeFieldPredicate, departement), 
              SimplePredicate: (commonThreeTokens, full_name), 
              SimplePredicate: (sortedAcronym, localite_acheminement_uai), 
              SimplePredicate: (commonFourGram, localite_acheminement_uai), 
              SimplePredicate: (wholeFieldPredicate, localite_acheminement_uai), 
              SimplePredicate: (sameSevenCharStartPredicate, localite_acheminement_uai), 
              SimplePredicate: (firstTokenPredicate, localite_acheminement_uai), 
              SimplePredicate: (nearIntegersPredicate, departement), 
              SimplePredicate: (firstIntegerPredicate, departement), 
              SimplePredicate: (firstIntegerPredicate, full_name), 
              SimplePredicate: (oneGramFingerprint, localite_acheminement_uai), 
              SimplePredicate: (metaphoneToken, full_name), 
              SimplePredicate: (sameSevenCharStartPredicate, full_name), 
              SimplePredicate: (doubleMetaphone, departement), 
              SimplePredicate: (firstTokenPredicate, full_name), 
              SimplePredicate: (twoGramFingerprint, full_name), 
              SimplePredicate: (tokenFieldPredicate, full_name), 
              SimplePredicate: (sortedAcronym, full_name), 
              SimplePredicate: (sameSevenCharStartPredicate, departement), 
              SimplePredicate: (suffixArray, localite_acheminement_uai), 
              SimplePredicate: (firstIntegerPredicate, localite_acheminement_uai), 
              SimplePredicate: (commonSixGram, full_name), 
              SimplePredicate: (twoGramFingerprint, departement), 
              SimplePredicate: (twoGramFingerprint, localite_acheminement_uai), 
              SimplePredicate: (sameThreeCharStartPredicate, full_name), 
              SimplePredicate: (commonTwoTokens, localite_acheminement_uai), 
              SimplePredicate: (firstTokenPredicate, departement), 
              SimplePredicate: (sameFiveCharStartPredicate, full_name), 
              SimplePredicate: (commonFourGram, full_name), 
              SimplePredicate: (commonTwoTokens, departement), 
              SimplePredicate: (commonSixGram, localite_acheminement_uai), 
              SimplePredicate: (commonFourGram, departement), 
              SimplePredicate: (sameThreeCharStartPredicate, departement), 
              SimplePredicate: (nearIntegersPredicate, full_name), 
              SimplePredicate: (doubleMetaphone, full_name), 
              SimplePredicate: (wholeFieldPredicate, full_name), 
              SimplePredicate: (fingerprint, full_name), 
              SimplePredicate: (sameThreeCharStartPredicate, localite_acheminement_uai), 
              SimplePredicate: (suffixArray, departement), 
              SimplePredicate: (doubleMetaphone, localite_acheminement_uai), 
              SimplePredicate: (commonTwoTokens, full_name)]

blocked_sample = {(689, 4275), (235, 6462), (1329, 1767), (1039, 2347), 
                  (46, 3905), (791, 6005), (729, 6836), (147, 3745),
                  (676, 5812), (240, 4294), (167, 5743), (1227, 4867), 
                  (419, 4062), (803, 4501), (842, 6036), (560, 3501), 
                  (521, 3404), (488, 4345), (43, 2032), (338, 1913), 
                  (787, 7185), (866, 5342), (1202, 3386), (110, 2753), 
                  (508, 4526), (1176, 7018), (1113, 5918), (789, 5687), (480, 7122)}

linkBlockedSample = functools.partial(blockedSample, linkSamplePredicates) 

def blockedSample(sampler, sample_size, predicates, *args):
    '''  '''
    
def linkSamplePredicates(sample_size, predicates, items1, items2):
    '''Yields linkSamplePredicate with items1 and items2 in randomized order'''
    items1 : len same as num rows in source
    items1[0] : (570, {'departement': '95', 'full_name': 'lycee georges braque', 'localite_acheminement_uai': 'argenteuil'})


def linkSamplePredicate(subsample_size, predicate, items1, items2):
    

evenSplits(b, a): basically yields b/a, a times 


/// Training
in learn()
len(dupe_cover) --> 2277
dupe_cover = {(LevenshteinSearchPredicate: (3, full_name), TfidfNGramSearchPredicate: (0.4, full_name)): {5}, (SimplePredicate: (sameSevenCharStartPredicate, full_name), SimplePredicate: (twoGramFingerprint, localite_acheminement_uai)): {27, 5}, (SimplePredicate: (firstTokenPredicate, localite_acheminement_uai), TfidfNGramSearchPredicate: (0.2, departement)): {0, 1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28}, TfidfNGramSearchPredicate: (0.4, full_name): {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}, (SimplePredicate: (commonFourGram, full_name), TfidfNGramSearchPredicate: (0.2, full_name)): {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}, (LevenshteinSearchPredicate: (1, departement), SimplePredicate: (twoGramFingerprint, full_name)): {5}, (LevenshteinSearchPredicate: (3, departement), SimplePredicate: (oneGramFingerprint, departement)): {3, 9, 12, 19, 21, 22}, (LevenshteinSearchPredicate: (2, departement)}

# Score for each blocking predicate
comparison_count = {(SimplePredicate: (metaphoneToken, departement), SimplePredicate: (metaphoneToken, localite_acheminement_uai)): 0.0, (LevenshteinSearchPredicate: (1, full_name), SimplePredicate: (commonIntegerPredicate, full_name)): 0.0, TfidfTextSearchPredicate: (0.6, departement): 7849.3968000000004, TfidfNGramSearchPredicate: (0.4, full_name): 300021.38879999996, (SimplePredicate: (commonFourGram, full_name), TfidfNGramSearchPredicate: (0.2, full_name)): 1425973.7519999999, (SimplePredicate: (nearIntegersPredicate, localite_acheminement_uai), SimplePredicate: (tokenFieldPredicate, full_name)): 0.0, SimplePredicate: (twoGramFingerprint, localite_acheminement_uai): 32269.742400000003, (SimplePredicate: (firstIntegerPredicate, localite_acheminement_uai), SimplePredicate: (nearIntegersPredicate, departement)): 0.0, (SimplePredicate: (tokenFieldPredicate, departement), TfidfNGramSearchPredicate: (0.6, full_name)): 872.15520000000004, (LevenshteinSearchPredicate: (2, localite_acheminement_uai), SimplePredicate: (firstIntegerPredicate, full_name)): 0.0, (SimplePredicate: (commonIntegerPredicate, localite_acheminement_uai), SimplePredicate: (suffixArray, localite_acheminement_uai)): 0.0,}


"""