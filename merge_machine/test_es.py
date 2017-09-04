#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:42:41 2017

@author: m75380

# Ideas: Learn analysers and weights for blocking on ES directly
# Put all fields to learn blocking by exact match on other fields

https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html
"""

import pandas as pd

from es_match import es_linker, Labeller


dir_path = 'data/sirene'
chunksize = 3000
file_len = 10*10**6


test_num = 0
if test_num == 0:
    source_file_path = 'local_test_data/source.csv'
    match_cols = [{'source': 'commune', 'ref': 'LIBCOM'},
                  {'source': 'lycees_sources', 'ref': 'NOMEN_LONG'}]    
    source_sep = ','
    source_encoding = 'utf-8'
    
elif test_num == 1:
    source_file_path = 'local_test_data/integration_5/data_ugly.csv'
    match_cols = [{'source': 'VILLE', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ETABLISSEMENT', 'ref': 'NOMEN_LONG'}]
    source_sep = ';'
    source_encoding = 'windows-1252'
    
elif test_num == 2:
    source_file_path = 'local_test_data/integration_3/export_alimconfiance.csv'
    match_cols = [{'source': 'Libelle_commune', 'ref': 'LIBCOM'},
                  #{'source': 'Libelle_commune', 'ref': 'L6_NORMALISEE'},
                  {'source': 'ods_adresse', 'ref': 'L4_NORMALISEE'},
                  {'source': 'APP_Libelle_etablissement', 'ref': ('L1_NORMALISEE', 
                                                        'ENSEIGNE', 'NOMEN_LONG')}]

    source_sep = ';'
    source_encoding = 'utf-8'
else:
    raise Exception('Not a valid test number')


source = pd.read_csv(source_file_path, 
                    sep=source_sep, encoding=source_encoding,
                    dtype=str, nrows=chunksize)
source = source.where(source.notnull(), '')

ref_table_name = '123vivalalgerie'




columns_to_index = {
    'SIREN': {},
    'NIC': {},
    'L1_NORMALISEE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'L4_NORMALISEE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'L6_NORMALISEE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'L1_DECLAREE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'L4_DECLAREE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'L6_DECLAREE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'LIBCOM': {
        'french', 'whitespace', 'end_n_grams', 'n_grams'
    },
    'CEDEX': {},
    'ENSEIGNE': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    'NOMEN_LONG': {
        'french', 'whitespace', 'integers', 'end_n_grams', 'n_grams'
    },
    #Keyword only 'LIBNATETAB': {},
    'LIBAPET': {},
    'PRODEN': {},
    'PRODET': {}
}

labeller = Labeller(source, ref_table_name, match_cols, columns_to_index)


#labeller.update_musts({'NOMEN_LONG': ['lycee']},
#                      {'NOMEN_LONG': ['ass', 'association', 'sportive', 'foyer']})

for i in range(100):
    (source_item, ref_item) = labeller.new_label()
    if not ref_item:
        print('No more examples to label')
        break
    
    for x in range(10):
        user_input = labeller._user_input(source_item, ref_item, test_num)
        if labeller.answer_is_valid(user_input):
            break
    else:
        raise ValueError('No valid answer after 10 iterations')
           
    is_match = labeller.parse_valid_answer(user_input)
    labeller.update(is_match, ref_item['_id'])
    
    if (test_num == 0) and i == 3:
        labeller.update_musts({'NOMEN_LONG': ['lycee']},
                              {'NOMEN_LONG': ['ass', 'association', 'sportive', 'foyer']})
        labeller.re_score_history()

print('best_query:\n', labeller.best_query_template())
print('must:\n', labeller.must)
print('must_not:\n', labeller.must_not)

assert False

# =============================================================================
# Match
# =============================================================================

# TODO: bool levels for fields also (L4 for example)

# query_template is ((source_key, ref_key, analyzer_suffix, boost), ...)

max_num_levels = 3 # Number of match clauses
bool_levels = {'.integers': ['must', 'should']}
#len(query_metrics[list(query_metrics.keys())[0]])
boost_levels = [1]

all_query_templates = gen_all_query_templates(match_cols, columns_to_index, 
                                              bool_levels, boost_levels, max_num_levels)

query_metrics = dict()
for q_t in all_query_templates:
    query_metrics[q_t] = []

import random

num_results_labelling = 3

num_found = 0
if test_num == 0:
    num_samples = 0
else:
    num_samples = 100
    
min_precision_tab = [(20, 0.7), (10, 0.5), (5, 0.3)] # (num_rows_labelled, min_prec)    

sorted_keys = all_query_templates

for num_rows_labelled, idx in enumerate(random.sample(list(source.index), num_samples)):
    row = source.loc[idx]
    print('Doing', row)  
        
    # Search elasticsearch for all results
    all_search_templates, full_responses = perform_queries(sorted_keys, [row], num_results_labelling)
    full_responses = {all_search_templates[idx][1][0]: values for idx, values in full_responses.items()}
    
    # Sort keys by score or most promising
    use_precision = num_found > 2
    if not use_precision:
        sorted_keys = random.sample(list(full_responses.keys()), len(full_responses.keys()))
        # sorted_keys = sorted(full_responses.keys(), key=lambda x: full_responses[x]['hits'].get('max_score') or 0, reverse=True)
    else:
        print('Using best precision')
        if found:
            agg_query_metrics = calc_agg_query_metrics(query_metrics)
            
        # Sort by ratio but label by precision ?
        sorted_keys = sorted(full_responses.keys(), key=lambda x: agg_query_metrics[x]['precision'], reverse=True)
        
        # Remove queries if precision is too low (thresh depends on number of labels)
        min_precision = 0
        for min_idx, min_precision in min_precision_tab:
            if num_rows_labelled >= min_idx:
                break
        sorted_keys = list(filter(lambda x: agg_query_metrics[x]['precision'] >= min_precision, sorted_keys))
    print('Number of keys left', len(sorted_keys))
    
    # Make user label data until the match is found
    found, res = gen_label(full_responses, sorted_keys, row, num_results_labelling, num_rows_labelled)









    

        #        import pdb
        #        pdb.set_trace()
        
# TODO: thresholder

# Sort by max score?

# Propose by descending score ?
# Alternate between my precision score and ES max score (or combination)

# TODO: only on chunksize rows of source

if test_num == 0:
    best_query_template = (('must', 'commune', 'LIBCOM', '.french', 1),
     ('must', 'lycees_sources', 'NOMEN_LONG', '.french', 10))
else:
    best_query_template = sorted(full_responses.keys(), key=lambda x: agg_query_metrics[x]['ratio'], reverse=True)[0]


params = {'query': best_query_template, 'thresh': 6}
new_source = es_linker(source, best_query_template, 6)# agg_query_metrics[best_query_template]['thresh'])
new_source['has_match'] = new_source.__CONFIDENCE.notnull()

if test_num == 2:
    new_source['good'] = new_source.SIRET == (new_source.SIREN + new_source.NIC)
    print('Precision:', new_source.loc[new_source.has_match, 'good'].mean())
    print('Recall:', new_source.good.mean())

# Display results
new_source.sort_values('__CONFIDENCE', inplace=True)

for i, row in new_source.iloc[:100].iterrows(): 
    print('**** ({0}) / score: {1}\n'.format(i, row['__CONFIDENCE']))
    for match in match_cols:
        if isinstance(match['ref'], str):
            cols = [match['ref']]
        else:
            cols = match['ref']
        for col in cols:
            print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
            print('> {1}   -> [{0}]'.format(col, row[col]))  
        
if test_num == 2:
    for i, row in new_source[new_source.has_match & ~new_source.good].iloc[:100].iterrows(): 
        print('**** ({0}) / score: {1}\n'.format(i, row['__CONFIDENCE']))
        for match in match_cols + [{'source': 'SIRET', 'ref': ('SIREN', 'NIC')}]:
            if isinstance(match['ref'], str):
                cols = [match['ref']]
            else:
                cols = match['ref']
            for col in cols:
                print('\n{1}   -> [{0}][source]'.format(match['source'], row[match['source']]))
                print('> {1}   -> [{0}]'.format(col, row[col]))  
#            
#        for match in match_cols:
#            print(row[match['source']], '\n', row[match['ref']], '\n')

assert False

# Number of results
len(query_metrics[list(query_metrics.keys())[0]])

# Sorted metrics
for x in sorted_keys[:40]:
    print(x, '\n -> ', agg_query_metrics[x], '\n')

t_ps = [x/400. for x in range(4*80, 4*100)]
precs = []
recs = []
ratios = []

for t_p in t_ps:
    tab = pd.DataFrame([(key, *compute_threshold(metrics, t_p=t_p, t_r=0)) for key, metrics in query_metrics.items() if key in sorted_keys])
    tab.columns = ['key', 'thresh', 'precision', 'recall', 'ratio']
    tab.sort_values('ratio', ascending=False, inplace=True)
    precs.append(tab.precision.iloc[0])
    recs.append(tab.recall.iloc[0])
    ratios.append(tab.ratio.iloc[0])
    
import matplotlib.pyplot as plt
plt.plot(t_ps, precs)
plt.plot(t_ps, recs)
plt.plot(t_ps, ratios)
print(tab.head(10))

'''
pre-process:
COMMUNE: remove "commune de"

pre-process: L1 + ENSEIGNE + NOMEN_LONG

Redresser les noms propres A. Camuse

tags pour les LIBAPET sirene ?

'''


'''
DATA?
Rungis   -> [Libelle_commune][source]
> CHEVILLY LARUE   -> [LIBCOM]
BAT E4 DU MIN DE PARIS RUNGIS   -> [ods_adresse][source]
> 2 AVENUE DE FLANDRE   -> [L4_NORMALISEE]
PALIMEX   -> [APP_Libelle_etablissement][source]
> PALIMEX   -> [L1_NORMALISEE]
330177635 00050 [source]
330177635 00050 [ref]

Marseille 7e  Arrondissement   -> [Libelle_commune][source]
> CHALLES LES EAUX   -> [LIBCOM]
LA LICORNE - PIZZERIA   -> [ods_adresse][source]
> 72 ALL DE LA BREISSE - RESIDENCE   -> [L4_NORMALISEE]
LA LICORNE   -> [APP_Libelle_etablissement][source]
> LGDI   -> [L1_NORMALISEE]
LA LICORNE   -> [APP_Libelle_etablissement][source]
> None   -> [ENSEIGNE]
822461620 00012 [source]
823997358 00010 [ref]

sans paramêtrage: Lycée Maxence Van der Meersch 
 ASS MAXENCE VAN DER MEERSCH 

Paris 14e  Arrondissement   -> [Libelle_commune][source]
> PARIS 14   -> [LIBCOM]

181 RUE DU CHATEAU   -> [ods_adresse][source]
> 171 RUE DU CHATEAU   -> [L4_NORMALISEE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> L ASSIETTE   -> [L1_NORMALISEE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> L'ASSIETTE   -> [ENSEIGNE]

L'ASSIETTE   -> [APP_Libelle_etablissement][source]
> NUAGE SARL   -> [NOMEN_LONG]
505240614 00014 [source]
505240614 00014 [ref]

Take queries with better precision first

'''
