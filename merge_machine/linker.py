#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo


# TODO: 
    - Generate variable definition. In particular: has missing/corpus/type
    - Translate all messages for the interface
    - Insert referential restriction in middle of training somehow

"""
import copy
import dedupe
import gc
import os
import re

import pandas as pd

# Dedupe data format: {rec_id_a: {field_1: val_1a, field_2: val_2a}, 
#                      rec_id_b: {field_1: val_1b, field_2: val_2b}}

# variable_definition: see here: https://dedupe.readthedocs.io/en/latest/Variable-definition.html


def preProcess(val):
    """
    Do a little bit of data cleaning with the help of
    [AsciiDammit](https://github.com/tnajdek/ASCII--Dammit) and
    Regex. Things like casing, extra spaces, quotes and new lines can
    be ignored.
    """
    val = re.sub('  +', ' ', val)
    val = re.sub('\n', ' ', val)
    val = val.strip().strip('"').strip("'").lower().strip()
    if val == '' :
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




def merge_results(ref, source, matched_records, selected_columns_from_ref):
    '''
    Takes the output of of matched records and merges the ref and source files.
    The output is of the same shape as the input source.
    
    INPUT:
        - ref: pd.DataFrame (reference)
        - source: pd.DataFrame (source)
        - matched_records: output of gazetteer.match
        - selected_columns_from_source: list of columns in ref that we want to 
                                        include in the file we return

    '''
    assert selected_columns_from_ref # TODO: This should be done in checking input parameters
    
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
    cols_for_match = get_cols_for_match(my_variable_definition)
    for col in cols_for_match:
        sel = tab[col].notnull()
        tab.loc[sel, col] = tab.loc[sel, col].apply(preProcess)
    
    # Replace np.NaN by None
    tab = tab.where(tab.notnull(), None)

    # Put as dedupe input format
    data = tab[cols_for_match].to_dict('index')
    
    return data


def main_dedupe(data_ref, data_source, my_variable_definition, train_path):
    '''Generates matches records'''
    sample_size = 50000
    
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
    
    gazetteer.sample(data_1=nonexact_1, data_2=nonexact_2, sample_size=sample_size)
    
    # Read training
    use_training_cache = True
    try:
        if use_training_cache and os.path.isfile(train_path):
            with open(train_path) as f:
                gazetteer.readTraining(f)
    except:
        print('Unable to load training data...')
        pass
        
    
    # Add examples through manual labelling
    # TODO: remove this when we can train in interface
    manual_labelling = True
    
    if manual_labelling:
        dedupe.consoleLabel(gazetteer)
        
        # Write training
        with open(train_path, 'w') as w:
            gazetteer.writeTraining(w)
     
    # Train on labelled data
    # TODO: Load train data
    gazetteer.train(index_predicates=True) # TODO: look into memory usage of index_predicates
    
    # Index reference
    gazetteer.index(data=data_ref)
    
    # Compute threshold
    recall_weight = 1.5
    threshold = gazetteer.threshold(data_source, recall_weight=recall_weight)

    matched_records = gazetteer.match(data_source, threshold=threshold)
    
    return matched_records, threshold
    
 
def linker(params):
    '''
    Takes as inputs file paths and returnes the merge table as a pandas DataFrame
    '''
    ref_path = params['ref_path']
    source_path = params['source_path']    
    train_path = params['train_path']   
    learned_settings_path = params['learned_settings_path']   
    my_variable_definition = params['my_variable_definition']   
    
    # Put to dedupe input format
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype='unicode')
    data_ref = format_for_dedupe(ref, my_variable_definition, 'ref') 
    del ref # To save memory
    gc.collect()
    
    # Put to dedupe input format
    source = pd.read_csv(source_path, encoding='utf-8', dtype='unicode')
    data_source = format_for_dedupe(source, my_variable_definition, 'source')
    del source
    gc.collect()
    
    matched_records, threshold = main_dedupe(data_ref, data_source, my_variable_definition, train_path)
    
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype='unicode')
    source = pd.read_csv(source_path, encoding='utf-8', dtype='unicode')
    
    # Generate out file
    source = merge_results(ref, source, matched_records, selected_columns_from_ref)
        
    return source


if __name__ == '__main__':
    
    # Not same as dedupe
    my_variable_definition = [
                            {'field': 
                                    {'source': 'lycees_sources',
                                    'ref': 'full_name'}, 
                            'type': 'String', 
                            'crf':True, 
                            'missing_values':True},
                                
                            {'field': {'source': 'commune', 
                                       'ref': 'localite_acheminement_uai'}, 
                            'type': 'String', 
                            'crf': True, 
                            'missing_values':True}
                            ]

    # What columns in reference to include in output
    selected_columns_from_ref = ['numero_uai', 'patronyme_uai', 'localite_acheminement_uai']
    
    
    train_path = 'local_test_data/training.json'
    #==============================================================================
    # Paths to data
    #==============================================================================
    file_role = 'ref'
    file_name = file_role + '.csv'
    ref_path = os.path.join('local_test_data', file_name)
    
    
    #==============================================================================
    # # GET SOURCE DATA
    #==============================================================================
    file_role = 'source'
    file_name = file_role + '.csv'
    source_path = os.path.join('local_test_data', file_name)
    
    #==============================================================================
    # 
    #==============================================================================
    
    

    # Explore results
    match_rate = source.numero_uai.notnull().mean()
    print(threshold, '-->', match_rate)

    source.sort_values(by='__CONFIDENCE', inplace=True)
    cols = ['commune', 'localite_acheminement_uai', 'lycees_sources', 
            'patronyme_uai', '__CONFIDENCE']
        
#    cProfile.run('main()')
    import pdb
    pdb.set_trace()
