#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo


# TODO: 
    - Generate variable definition. In particular: has missing/corpus/type
    - Translate all messages for the interface
    - Insert referential restriction in middle of training somehow
    - Check before hand that the columns asked for exist
# ERRORS in DOC:
indexes=True --> index=True

Given a dedupe formated training set, perform linking between source and reference using gazetteer.

"""
import copy
import dedupe
import gc
import json
import os
import re
from string import punctuation

import pandas as pd
import unidecode

# Dedupe data format: {rec_id_a: {field_1: val_1a, field_2: val_2a}, 
#                      rec_id_b: {field_1: val_1b, field_2: val_2b}}

# variable_definition: see here: https://dedupe.readthedocs.io/en/latest/Variable-definition.html

NO_TRAINING_MESSAGE = 'No training file could be found. Use the interface (XXX)' \
                      ', specify a matching ID to generate train (YYY), or ' \
                      'upload a training file using (ZZZ)'

def pd_pre_process(series, remove_punctuation=False):
    '''Applies pre-processing to series using builtin pandas.str'''
    series = series.str.replace(' +', ' ')
    series = series.str.replace('\n', ' ')
    if remove_punctuation:
        for punc in punctuation:
            series = series.str.replace(punc, ' ')
    series = series.str.strip(' \"\'').str.lower()
    series = series.replace('', None)
    return series

def pre_process(val, remove_punctuation=False):
    """
    Do a little bit of data cleaning. Things like casing, extra spaces, 
    quotes and new lines can be ignored.
    
    TODO: parallelise
    """
    val = re.sub('  +', ' ', val)
    val = re.sub('\n', ' ', val)
    if remove_punctuation:
        for punc in punctuation:
            val = re.sub(re.escape(punc), ' ', val)
    val = val.strip(' \"\'').lower()
    val = unidecode.unidecode(val)
    if val == '':
        val = None
    return val

def exact_matches(data_1, data_2, match_fields):
    '''
    (from https://github.com/datamade/csvdedupe/blob/master/csvdedupe/csvlink.py)
    Separates dedupe-formated data from two sources into exact matches, and non-
    exact matches
    '''
    nonexact_1 = {}
    nonexact_2 = {}
    exact_pairs = []
    redundant = {}

    for key, record in data_1.items():
        record_hash = hash(tuple(record[f] for f in match_fields))
        redundant[record_hash] = key        

    for key_2, record in data_2.items():
        record_hash = hash(tuple(record[f] for f in match_fields))
        if record_hash in redundant:
            key_1 = redundant[record_hash]
            exact_pairs.append(((key_1, key_2), 1.0))
            del redundant[record_hash]
        else:
            nonexact_2[key_2] = record

    for key_1 in redundant.values():
        nonexact_1[key_1] = data_1[key_1]
        
    return nonexact_1, nonexact_2, exact_pairs


def merge_results(ref, source, matched_records):
    '''
    Takes the output of of matched records and merges the ref and source files.
    The output is of the same shape as the input source.
    
    INPUT:
        - ref: pd.DataFrame (reference)
        - source: pd.DataFrame (source)
        - matched_records: output of gazetteer.match
        - selected_columns_from_ref: list of columns in ref that we want to 
                                        include in the file we return
    '''
    selected_columns_from_ref = list(ref.columns)
    #    if selected_columns_from_ref is None:
    #        selected_columns_from_ref = list(ref.columns)
    #    assert selected_columns_from_ref # TODO: This should be done in checking input parameters
    
    # Turn matched_records into pandas DataFrame
    source_idx = [x[0][0][0] for x in matched_records]
    ref_idx = [x[0][0][1] for x in matched_records]
    confidence = [x[0][1] for x in matched_records]
    matched_records_df = pd.DataFrame(list(zip(source_idx, ref_idx, confidence)), 
                            columns = ['__SOURCE_IDX', '__REF_IDX', '__CONFIDENCE'])
    
    source = source.merge(matched_records_df, left_index=True, 
                          right_on='__SOURCE_IDX', how='left')
    source = source.merge(ref[selected_columns_from_ref], 
                          left_on = '__REF_IDX', right_index=True, how='left')
    
    for col in ['__SOURCE_IDX', '__REF_IDX']:
        source.drop(col, inplace=True, axis=1)

    return source

def get_cols_for_match(my_variable_definition):
    cols_for_match = [var['field']['ref'] for var in my_variable_definition]
    return cols_for_match

def format_for_dedupe(tab, my_variable_definition, file_role):
    '''Takes a pandas DataFrame and create a dedupe-compatible dictionnary'''
    # Check that all columns are destinct
    if len(set(tab.columns)) != tab.shape[1]:
        raise Exception('CSV inputs should have distinct column names')
    
    # Replace columns in source
    if file_role == 'source':
        col_map = {x['field']['source']: x['field']['ref'] for x in my_variable_definition}
        tab.columns = [col_map.get(x, x) for x in tab.columns]
    
    # Pre-process
    print('before preprocessing')
    cols_for_match = get_cols_for_match(my_variable_definition)
    for col in cols_for_match:
        print('At column', col)
        sel = tab[col].notnull()
        tab.loc[sel, col] = tab.loc[sel, col].apply(pre_process)
    print('after preprocessing')
    
    # Replace np.NaN by None
    tab = tab.where(tab.notnull(), None)

    # Put as dedupe input format
    data = tab[cols_for_match].to_dict('index')
    
    return data



def old_load_deduper(data_ref, data_source, my_variable_definition):
    '''Load the dedupe object; TODO: duplicate with code in main dedupe'''
    
    SAMPLE_SIZE = 5000
    
    # Get columns for match and change to standard dedupe variable definition
    cols_for_match = get_cols_for_match(my_variable_definition)
    variable_definition = copy.deepcopy(my_variable_definition)
    for var in variable_definition:
        var['field'] = var['field']['ref']    
    
    (nonexact_1,
     nonexact_2,
     exact_pairs) = exact_matches(data_ref, data_source, cols_for_match)

    num_cores = 2
    gazetteer = dedupe.Gazetteer(variable_definition=variable_definition, 
                                 num_cores=num_cores)
    
    gazetteer.sample(data_1=nonexact_1, data_2=nonexact_2, sample_size=SAMPLE_SIZE)
    
    return gazetteer



def main_dedupe(data_ref, 
                data_source, 
                my_variable_definition, 
                train_path, 
                learned_settings_path):
    '''Computes matches between records'''
    
    
    SAMPLE_SIZE = 5000
    
    # Get columns for match and change to standard dedupe variable definition
    cols_for_match = get_cols_for_match(my_variable_definition)
    variable_definition = copy.deepcopy(my_variable_definition)
    for var in variable_definition:
        var['field'] = var['field']['ref']    
    
    (nonexact_1,
     nonexact_2,
     exact_pairs) = exact_matches(data_ref, data_source, cols_for_match)

    num_cores = 2
    gazetteer = dedupe.Gazetteer(variable_definition=variable_definition, 
                                 num_cores=num_cores)    

    
    # Read training
    use_training_cache = True
    try:
        # Replace with this in future version
        #if not os.path.isfile(train_path):
        #    raise Exception(NO_TRAINING_MESSAGE)
        
        if use_training_cache and os.path.isfile(train_path):
            with open(train_path) as f:
                gazetteer.readTraining(f)
    except:
        print('Unable to load training data...')
        pass
        
    
    # Add examples through manual labelling
    # TODO: remove this when we can train in interface
    manual_labelling = False
    if manual_labelling:
        dedupe.consoleLabel(gazetteer)
        
        # Write training
        with open(train_path, 'w') as w:
            gazetteer.writeTraining(w)
     
    # Train on labelled data
    # TODO: Load train data
    all_predicates = []
    gazetteer.sample(data_1=nonexact_1, data_2=nonexact_2, sample_size=SAMPLE_SIZE)
    gazetteer.train(recall=0.95, index_predicates=True) # TODO: look into memory usage of index_predicates
    all_predicates.extend(list(gazetteer.predicates))
        
    import dedupe.blocking as blocking
    gazetteer.blocker = blocking.Blocker(all_predicates)
    gazetteer.predicates = tuple(all_predicates)
        #all_predicates.extend(gazetteer.predicates)

    # Write settings    
    with open(learned_settings_path, 'wb') as f: # TODO: replaced 'wb' by 'w'
        gazetteer.writeSettings(f, index=False)    
    
    # Index reference
    print('indexing')
    gazetteer.index(data=data_ref)
    print('done indexing')
    
    # Compute threshold
    recall_weight = 2.5
    threshold = gazetteer.threshold(data_source, recall_weight=recall_weight)
    print('Threshold', threshold)

    matched_records = gazetteer.match(data_source, threshold=threshold)
    return matched_records, threshold
    

# TODO: use trainingDataLink for automatic labelling when a common key is available
 
def dedupe_linker(paths, params):
    '''
    Takes as inputs file paths and returnes the merge table as a pandas DataFrame
    
    Sample arguments:
    
    paths={'ref': 'path_to_reference',
           'source': 'path_to_source',
           'train': 'path_to_training_file',
           'learned_settings': 'path_to_learned_settings'}
    params={'variable_definition': dict_of_variable_definitions TODO: add this,
            'selected_columns_from_ref': unknown: TODO: add this}
    '''
    ref_path = paths['ref']
    source_path = paths['source']    
    train_path = paths['train']   
    learned_settings_path = paths['learned_settings']
    my_variable_definition = params['variable_definition']
    
    # Put to dedupe input format
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype=str)
    data_ref = format_for_dedupe(ref, my_variable_definition, 'ref') 
    del ref # To save memory
    gc.collect()
    
    # Put to dedupe input format
    source = pd.read_csv(source_path, encoding='utf-8', dtype=str)
    data_source = format_for_dedupe(source, my_variable_definition, 'source')
    del source
    gc.collect()
    
    matched_records, threshold = main_dedupe(data_ref, 
                                             data_source, 
                                             my_variable_definition, 
                                             train_path,
                                             learned_settings_path)
    
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype=str)
    source = pd.read_csv(source_path, encoding='utf-8', dtype=str)
    
    # Generate out file
    source = merge_results(ref, source, matched_records)
    
    # Add  results of manual labelling
    #    if add_labels:
    #        labels = json.load(open(train_path))
    #        
    #        import pdb
    #        pdb.set_trace()
    
    # Run info
    run_info = {'threshold': threshold}
    
    return source, run_info


if __name__ == '__main__':    
    match_rates = []
#    for i in range(5):
    new_param = 200
    for i in range(5):
        with open('local_test_data/lycees/config.json') as f:
           my_config = json.load(f)    
        paths = my_config['paths']
        params = my_config['params']
        
        source, threshold = dedupe_linker(paths, params)
    
        source.to_csv('local_test_data/rnsr/res.csv', encoding='utf-8', index=False)
    
        # Explore results
        sel = source.__CONFIDENCE.notnull()
        match_rate = sel.mean()
        match_num = sel.sum()
        prec = (source.loc[sel, :].uai.str.upper() == source.loc[sel, :].numero_uai.str.upper()).mean()
        print('\n', i, '', threshold, '-->', match_rate, ' / ', match_num, ' / prec: ', prec)
        match_rates.append((match_rate, prec))
    
        source.sort_values(by='__CONFIDENCE', inplace=True)
        cols = ['commune', 'localite_acheminement_uai', 'lycees_sources', 
                'patronyme_uai', '__CONFIDENCE']
        
#    cProfile.run('main()')
