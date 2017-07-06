#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:13:02 2017
@author: leo

Try to infer properties of the reference file based on given matches (from manual 
training or exact match on a given field). In particular: guess words that 
SHOULD be present in a given column. You can then restrict the reference to 
lines that ALL contain these words.

For example: if source only contains companies listed in departement:"Essonne" 
in the reference for training matches, you can suggest that we should only 
search for matches within rows of the reference that have departement:"Essonne"

# TODO: Add threshold for value presence to account for user mistake

"""

import pandas as pd

#==============================================================================
# Generic functions to restrict reference
#==============================================================================

def find_common_words(ref_match, cols=None, pre_process_func=None):
    """
    Use on result of training and/or exact match to infer common words in 
    columns to restrict matching to a subset of the data. 
    
    INPUT:
        - ref_match: pandas DataFrame to consider
        - cols: (defaults to None) columns in which too look for common words
                None will use all columns except for "_has_match"
    
    OUTPUT: 
        - all_candidate_words: dict with key the column name and value a list
                               of words in common
        - pre_process_func: function that modifies takes a pd.Series and returns 
                clean pd.Series
    """
    if cols is None:
        cols = [x for x in ref_match.columns if x != '_has_match']
    
    all_candidate_words = dict()
    for col in cols:
        all_candidate_words[col] = _find_common_words_in_col(ref_match, col, pre_process_func)
    return all_candidate_words
   
def _find_common_words_in_col(ref_match, col, pre_process_func):
    """
    Finds words that are present in all values of the column specified by col.
    """
    if pre_process_func is not None:
        all_words = pd.Series(pre_process_func(ref_match[col]).str.cat(sep=' ').split()).value_counts()
    else:
        all_words = pd.Series(ref_match[col].str.cat(sep=' ').split()).value_counts()
    sel = all_words >= len(ref_match)
    common_words = []
    for word in all_words[sel].index:
        if ref_match[col].str.contains(word).all():
            common_words.append(word)
    return common_words    

def find_common_vals(ref_match, cols=None, pre_process_func=None):
    """
    Use on result of training and/or exact match to infer common exact values
    in columns to restrict matching to a subset of the data. 
    
    INPUT:
        - ref_match: pandas DataFrame to consider
        - cols: (defaults to None) columns in which too look for common words
                None will use all columns except for "_has_match"
        - pre_process_func: function that modifies takes a pd.Series and returns 
                clean pd.Series
    
    OUTPUT: 
        - all_candidate_words: dict with key the column name and value a list
                               of words in common
    """
    if cols is None:
        cols = [x for x in ref_match.columns if x != '_has_match']    

    all_candidate_values = dict()
    for col in cols:
        if (ref_match[col] == ref_match[col].iloc[0]).all():
            all_candidate_values[col] = ref_match[col].iloc[0]
        else:
            all_candidate_values[col] = None
    return all_candidate_values


def filter_by_words(ref, col_words, pre_process_func=None):
    """
    Filters rows in ref by words specified in col_words
    
    INPUT:
        - ref: pandas DataFrame
        - col_words: dictionnary mapping lists_of words to columns (same as 
                    output of find_common_words). Specifies the words to look 
                    for in each column (to keep).
        - pre_process_func: function that modifies takes a pd.Series and returns 
                clean pd.Series
    OUTPUT:
        - ref: table filtered by words to keep for each column
        - run_info: original table length and new length
    """
    
    run_info = {'og_len': len(ref)}
    run_info['has_modifications'] = False
    
    for col, words in col_words.items():
        for word in words:
            if pre_process_func is not None:
                ref = ref[pre_process_func(ref[col]).str.contains(word)]
            else:
                ref = ref[ref[col].str.contains(word)]
            run_info['has_modifications'] = True            
            
    run_info['new_len'] = len(ref)
    return ref, run_info


def filter_by_vals(ref, col_vals, pre_process_func=None):
    """
    Filters rows in ref by words specified in col_words
    
    INPUT:
        - ref: pandas DataFrame
        - col_vals: dictionnary mapping values to columns (same as 
                    output of find_common_vals). Specifies the value to look 
                    for in each column (to keep).
        - pre_process_func: function that modifies takes a pd.Series and returns 
                clean pd.Series                    
    OUTPUT:
        - ref: table filtered by words to keep for each column
        - run_info: original table length and new length
    """
    run_info = {'og_len': len(ref)}
    run_info['has_modifications'] = False
    
    for col, val in col_vals.items():
        if val is not None:
            if pre_process_func is not None:
                ref = ref[pre_process_func(ref[col]) == val]
            else:
                ref = ref[ref[col] == val]
            run_info['has_modifications'] = True 
            
    run_info['new_len'] = len(ref)
    return ref, run_info

#==============================================================================
# Project specific module
#==============================================================================

# TODO: dirty! move this
from dedupe_linker import pd_pre_process

def training_to_ref_df(training):
    '''
    Takes as input a dedupe training file and returns a pandas DataFrame with
    the data corresponding to match samples in the referential.
    '''
    training_df = pd.DataFrame([x['__value__'][1] for x in training['match']])
    return training_df

def infer_restriction(_, params):
    ''' #TODO: document '''
    training = params['training']
    filter_by_vals = params.get('filter_by_vals', True)
    filter_by_words = params.get('filter_by_words', True)

    if not training:
        raise ValueError('training has no values (minimum 1)')
    
    training_df = training_to_ref_df(training)
    
    to_return = dict()
    if filter_by_vals:
        to_return['col_words'] = find_common_words(training_df, pre_process_func=pd_pre_process)
    if filter_by_words:
        to_return['col_vals'] = find_common_vals(training_df, pre_process_func=pd_pre_process)
        
    to_return['filter_by_words'] = filter_by_words
    to_return['filter_by_vals'] = filter_by_vals

    return to_return

def perform_restriction(ref, params):
    ''' #TODO: document '''
    run_info = dict()
    
    if params['filter_by_vals']:
        ref, run_info['filter_by_vals'] = filter_by_vals(ref, params['col_vals'], pre_process_func=pd_pre_process)    
    
    if params['filter_by_words']:
        ref, run_info['filter_by_words'] = filter_by_words(ref, params['col_words'], pre_process_func=pd_pre_process)

    return ref, run_info

#def sample_restriction(ref, params, sample_params):
#    '''    
#    IN:
#        - tab: the pandas DataFrame from which to sample
#        - params: the parameters returned by infer_restriction
#        - sample_parameters: parameters contolling sample size etc.
#    '''
#    
#    TODO: This

if __name__ == '__main__':
    from linker import UserLinker
    
    project_id = '78246d462d500c1234903cc338c7c495'    
    proj = UserLinker(project_id)    
    training = proj.read_config_data('dedupe_linker', 'training.json')    
    
    training_df = training_to_ref_df(training)
    common_words = find_common_words(training_df)
    common_vals = find_common_vals(training_df)
    
    params = infer_restriction(None, {'training': training})