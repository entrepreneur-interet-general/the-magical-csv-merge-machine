#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo


# TODO: 
    - Generate variable definition. In particular: has missing/corpus/type
    - Translate all messages for the interface

"""
import dedupe
import os
import re

import pandas as pd

# Dedupe data format: {rec_id_a: {field_1: val_1a, field_2: val_2a}, 
#                      rec_id_b: {field_1: val_1b, field_2: val_2b}}

# variable_definition: see here: https://dedupe.readthedocs.io/en/latest/Variable-definition.html


def index_col_map(col_map, file_role):
    '''
    Returns a dict with keys the value of file_role and value the new column 
    name
    '''
    assert file_role in ['source', 'ref']
    new_col_map = {_map[file_role]: _map['col_new'] for _map in col_map}
    return new_col_map

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


if __name__ == '__main__':
    
    # Dict that maps a new nam
    col_map = [
        {'source': 'lycees_sources','ref': 'full_name', 'col_new': 'nom_lycee'},
        {'source': 'commune', 'ref': 'localite_acheminement_uai', 'col_new': 'commune'}
        ]
    
    
    variable_definition = [
                            {'field': 'nom_lycee', 'type': 'String'},
                            {'field': 'commune', 'type': 'String'}
                            ]
    cols = [var['field'] for var in variable_definition]
    
    num_cores = 2
    gazetteer = dedupe.Gazetteer(variable_definition=variable_definition, 
                                 num_cores=num_cores)
    
    
    train_path = 'local_test_data/training.json'
    
    
    
    #==============================================================================
    # # GET REF DATA
    #==============================================================================
    file_role = 'ref'
    file_name = file_role + '.csv'
    ref_path = os.path.join('local_test_data', file_name)
    
    tab = pd.read_csv(ref_path, encoding='utf-8', dtype='unicode')
    
    # Check that all columns are destinct
    if len(set(tab.columns)) != tab.shape[1]:
        raise Exception('CSV inputs should have distinct column names')
    
    # Replace columns
    indexed_col_map = index_col_map(col_map, file_role)
    tab.columns = [indexed_col_map.get(x, x) for x in tab.columns]
    
    # Pre-process
    for col in cols:
        sel = tab[col].notnull()
        tab.loc[sel, col] = tab.loc[sel, col].apply(preProcess)
    
    # Replace np.NaN by None
    tab = tab.where(tab.notnull(), None)
    
    # Put to dedupe input format
    data_ref = tab[cols].to_dict('index')
    
    # Remove np.nan
    #data_ref = {_id: {key: value for key, value in datapoint.iteritems() if isinstance(value, basestring)} for _id, datapoint in data_ref.iteritems()}
    
    #==============================================================================
    # # GET SOURCE DATA
    #==============================================================================
    file_role = 'source'
    file_name = file_role + '.csv'
    source_path = os.path.join('local_test_data', file_name)
    
    tab = pd.read_csv(source_path, encoding='utf-8', dtype='unicode')
    
    # Check that all columns are destinct
    if len(set(tab.columns)) != tab.shape[1]:
        raise Exception('CSV inputs should have distinct column names')
    
    # Replace columns
    indexed_col_map = index_col_map(col_map, file_role)
    tab.columns = [indexed_col_map.get(x, x) for x in tab.columns]
    
    # Pre-process
    for col in cols:
        sel = tab[col].notnull()
        tab.loc[sel, col] = tab.loc[sel, col].apply(preProcess)
    
    # Replace np.NaN by None
    tab = tab.where(tab.notnull(), None)
    
    
    # Remove np.nan
    #data_ref = {_id: {key: value for key, value in datapoint.iteritems() if isinstance(value, basestring)} for _id, datapoint in data_ref.iteritems()}
    
    # Put to dedupe input format
    data_source = tab[cols].to_dict('index')
    
    #==============================================================================
    # 
    #==============================================================================
    
    # Manual train # TODO: remove
    sample_size = 500000
    
    (nonexact_1,
     nonexact_2,
     exact_pairs) = exact_matches(data_ref, data_source, cols)
    
    gazetteer.sample(data_1=nonexact_1, data_2=nonexact_2, sample_size=sample_size)
    
    # Read training
    if os.path.isfile(train_path):
        with open(train_path) as f:
            gazetteer.readTraining(f)
    
    # Add training
    dedupe.consoleLabel(gazetteer)
    # Write training
    with open(train_path, 'w') as w:
        gazetteer.writeTraining(w)
     
    # Add training
    max_compare = 500000
    gazetteer.train(index_predicates=True)
    
    
    # Index reference
    gazetteer.index(data=data_ref)
    
    threshold = 0.0001
    
    matched_records = gazetteer.match(data_source, threshold=threshold)
    
    
    
    
    # Merge source with selected columns in ref
    selected_columns_from_ref = ['numero_uai', 'patronyme_uai', 'localite_acheminement_uai']
    
    source = pd.read_csv(source_path, encoding='utf-8', dtype='unicode')
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype='unicode')
    
    
    source = merge_results(ref, source, matched_records, selected_columns_from_ref)
    
    
    
    
    
    import pdb
    pdb.set_trace()
