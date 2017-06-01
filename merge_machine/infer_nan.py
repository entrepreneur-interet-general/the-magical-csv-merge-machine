#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:11:07 2017

@author: leo

Take table data and infer values that represent missing data to replace by 
standard values for missing data.

Examples of missing data that can be found:
    - 'XXX'
    - 'NO_ADDR'
    - '999999'
    - 'NONE'
    - '-' 
    
TODO:
    - For probable missing values, check entire file
"""

import string
import numpy as np
import pandas as pd

def mv_from_letter_repetition(top_values, score=0.7):
    """Checks for unusual repetition of characters as in XX or 999999"""
    # Compute number of unique characters for each value
    num_unique_chars = pd.Series([len(set(list(x))) for x in top_values.index], index=top_values.index)
    # Check that we have at least 3 distinct values
    if len(num_unique_chars) >= 3:
        # Keep as NaN candidate if value has only 1 unique character and 
        # at least two characters and other values have at least two disctinct
        # characters
        num_unique_chars.sort_values(inplace=True)
        if (len(num_unique_chars.index[0]) > 1) \
                and (num_unique_chars.iloc[0] == 1) \
                and (num_unique_chars.iloc[1] >= 2):
            return [(num_unique_chars.index[0], score, 'letter_repeted')]
    return []

def mv_from_usual_forms(top_values, probable_missing_values, score=0.5):
    """Compares top values to common expressions for missing values"""
    to_return = []
    for val in top_values.index:
        if val.lower() in [x.lower() for x in probable_missing_values]:
            to_return.append((val, score, 'usual'))
    return to_return
    

def mv_from_len_diff(top_values, score=0.3):
    """Check if all values have the same length except one"""
    # Compute lengths of values
    lengths = pd.Series([len(x) for x in top_values.index], index=top_values.index)
    # Check if all values have the same length except one:
    if (lengths.nunique() == 2) & (len(top_values) >= 4):
        
        if lengths.value_counts().iloc[-1] == 1:
            abnormal_length = lengths.value_counts().index[-1]
            mv_value = lengths[lengths == abnormal_length].index[0]
            return [(mv_value, score, 'diff')]
    return []

def mv_from_len_ratio(top_values, score=0.2):
    """Check if one value is much shorter than others"""
    # Compute lengths of values
    lengths = pd.Series([len(x) for x in top_values.index], index=top_values.index)
    
    if len(top_values) >= 4:
        lengths.sort_values(inplace=True)
        length_ratio = 2.9
        if length_ratio * lengths.iloc[0] < lengths.iloc[1]:
            mv_value = lengths.index[0]
            return [(mv_value, score, 'len_ratio')]
    return []


def mv_from_not_digit(top_values, score=1):
    """Check if one value is the only not digit"""
    is_digit = pd.Series([x.replace(',', '').replace('.', '').isdigit() 
                        for x in top_values.index], index=top_values.index)
    if len(top_values) >= 3:
        if (~is_digit).sum() == 1:
            mv_value = is_digit[is_digit == False].index[0]
            return [(mv_value, score/2. + score/2.*(len(top_values) >= 4), 'not_digit')]
    return []


def mv_from_punctuation(top_values, score=1):
    """Check if value is only one with only punctuation"""    
    punct = string.punctuation + ' '
    is_punct = pd.Series([all(y in punct for y in x) for x in top_values.index], index=top_values.index)
    if (is_punct).sum() == 1:
        mv_value = is_punct[is_punct].index[0]
        return [(mv_value, score, 'punctuation')]
    return []

def mv_from_common_values(all_top_values, score=0.5):
    '''Looks for values common in at least two columns'''
    # Create dict with: {value: set_of_columns_where_common} with values present in at least two columns
    popular_values = dict()
    for col_1, top_values_1 in all_top_values.items():
        for col_2, top_values_2 in all_top_values.items():
            if col_1 != col_2:
                common_values = [x for x in top_values_1.index if x in top_values_2.index]
                for val in common_values:
                    if val not in popular_values:
                        popular_values[val] = set([col_1, col_2])
                    else:
                        popular_values[val].add(col_1)
                        popular_values[val].add(col_2)
    
    if popular_values:
        # Questionable heuristic: return value most frequent
        temp = [(val, len(cols)) for val, cols in popular_values.items()]
        temp.sort(key=lambda x: x[1], reverse=True)
        mv_value = temp[0][0]
        return [(mv_value, score, 'common_values')]
    return  []
                   

def mv_from_common_values_2(col_mvs, score=1):
    """
    Return mv candidates for missing values that are already candidates in 
    at least two columns.    
    """
    # Make dict with key: mv_candidate value: list of columns where applicable
    val_mvs = dict()
    for col, tuples in col_mvs.items():
        for (val, score, origin) in tuples:
            if val not in val_mvs:
                val_mvs[val] = [col]
            else:
                val_mvs[val].append(col)
                
    return [(val, score, 'common_values') for val, cols in val_mvs.items() if (len(cols)>=2)]
    

def compute_all_top_values(tab, num_top_values):
    '''
    Returns a dict with columns of the table as keys and the top 10 most 
    frequent values in a pandas Series
    '''
    all_top_values = dict()
    for col in tab.columns:
        all_top_values[col] = tab[col].value_counts(True).head(num_top_values)
    return all_top_values



def correct_score(list_of_possible_mvs, probable_mvs):
    """
    Corrects original scores by comparing string distance to probable_mvs
    
    INPUT:
        list_of_possible_mvs: ex: [(mv, 0.3), (branch, 0.2)]
        probable_mvs: ex ['nan', 'none']
    OUTPUT:
        list_of_possible_mvs: ex[(nan, 0.9), (branch, 0.1)]
    """
    # Sum scores for same values detected by different methods in same column
    new_list_of_possible_mvs_tmp = dict()
    for (val, coef, orig) in list_of_possible_mvs:
        if val not in new_list_of_possible_mvs_tmp:
            new_list_of_possible_mvs_tmp[val] = dict()
            new_list_of_possible_mvs_tmp[val]['score'] = coef
            new_list_of_possible_mvs_tmp[val]['origin'] = [orig] 
        else:
            new_list_of_possible_mvs_tmp[val]['score'] += coef
            new_list_of_possible_mvs_tmp[val]['origin'].append(orig)                           
                                    
    # NB: Taken care of in mv_from_usual_forms
    #        # If the value is a known form of mv, increase probability 
    #        if val.lower() in [x.lower() for x in probable_mvs]:
    #            new_list_of_possible_mvs_tmp[val] += 0.5

    # Reformat output like input
    new_list_of_possible_mvs = []
    for val, _dict in new_list_of_possible_mvs_tmp.items():
        new_list_of_possible_mvs.append((val, _dict['score'], _dict['origin']))
    
    return new_list_of_possible_mvs


def infer_mvs(tab, params=None):
    """
    API MODULE
    
    Run mv inference processes for each column and for the entire table
    """
    PROBABLE_MVS = ['nan', 'none', 'na', 'n/a', '\\n', ' ', 'non renseigne', \
                    'nr', 'no value', 'null', 'missing value']
    ALWAYS_MVS = ['']
    
    if params is None:
        params = {}
    
    # Set variables and replace by default values
    PROBABLE_MVS.extend(params.get('probable_mvs', []))
    ALWAYS_MVS.extend(params.get('always_mvs', []))
    num_top_values = params.get('num_top_values', 10)
    
    # Compute most frequent values per column
    all_top_values = compute_all_top_values(tab, num_top_values)
    
    col_mvs = dict()
    # Look at each column and infer mv
    for col, top_values in all_top_values.items():
        col_mvs[col] = []
        if top_values.iloc[0] == 1:
            continue
        col_mvs[col].extend(mv_from_len_diff(top_values))
        col_mvs[col].extend(mv_from_len_ratio(top_values))
        col_mvs[col].extend(mv_from_not_digit(top_values))
        col_mvs[col].extend(mv_from_punctuation(top_values))
        col_mvs[col].extend(mv_from_usual_forms(top_values, PROBABLE_MVS))
        col_mvs[col].extend(mv_from_usual_forms(top_values, ALWAYS_MVS, 10**3))
        col_mvs[col].extend(mv_from_letter_repetition(top_values))
        col_mvs[col] = correct_score(col_mvs[col], PROBABLE_MVS)
        col_mvs[col].sort(key=lambda x: x[1], reverse=True)


    # Transfer output to satisfy API standards
    def triplet_to_dict(val):
        return {'val': val[0], 'score': val[1], 'origin': val[2]}

    common_mvs = [triplet_to_dict(val) for val in mv_from_common_values_2(col_mvs)]
    columns_mvs = {key:[triplet_to_dict(val) for val in vals] for key, vals in col_mvs.items() if vals}
    infered_mvs = {'columns': columns_mvs, 'all': common_mvs}
    return {'mvs_dict': infered_mvs, 'thresh': 0.6} # TODO: remove harcode

def replace_mvs(tab, params):
    """
    API MODULE
    
    Replace the values that should be mvs by actual np.nan. Values in 'all' 
    will be replaced in the entire table whereas values in 'columns' will only
    be replaced in the specified columns.
    
    INPUT:
        tab: pandas DataFrame to modify
        OBSOLETE
        mvs_dict: dict indicating mv values with scores. For example:
                {
                    'all': [],
                    'columns': {'dech': [('-', 2.0, 'unknown')],
                                'distance': [('-', 1, 'unknown')]}
                }
        thresh: minimum score to remove mvs
    
    OUTPUT:
        tab: same table with values replaced by np.nan
        run_info    
    """
    DEFAULT_THRESH = 0.6
    
    # Set variables and replace by default values
    mvs_dict = params['mvs_dict']
    thresh = params.get('thresh', DEFAULT_THRESH)
    
    # Replace
    assert sorted(list(mvs_dict.keys())) == ['all', 'columns']

    # Run information
    run_info = dict()
    run_info['modified_columns'] = []
    run_info['has_modifications'] = False
    run_info['replace_num'] = {'all': dict(), 'columns': dict()}

    for mv in mvs_dict['all']:
        val, score = mv['val'], mv['score']
        run_info['replace_num']['all'][val] = 0
        if score >= thresh:
            # Metrics
            col_count = (tab == val).sum()
            run_info['modified_columns'] = run_info['modified_columns'].extend(list(col_count[col_count != 0].index))
            run_info['has_modifications'] = run_info['has_modifications'] or (col_count.sum() >= 1)
            run_info['replace_num']['all'][val] = int(col_count.sum())
            
            # Do transformation
            tab.replace(val, np.nan, inplace=True)
    
    for col, mv_values in mvs_dict['columns'].items():
        run_info['replace_num']['columns'][col] = dict()
        for mv in mv_values:
            val, score = mv['val'], mv['score']
            run_info['replace_num']['columns'][col][val] = 0
            if score >= thresh:
                # Metrics
                count = (tab[col] == val).sum()
                if count:
                    run_info['modified_columns'].append(col)
                    run_info['has_modifications'] = run_info['has_modifications'] or (count >= 1)
                run_info['replace_num']['columns'][col][val] = int(count)      
                
                # Do transformation
                tab[col].replace(val, np.nan, inplace=True)

    run_info['modified_columns'] = list(set(run_info['modified_columns']))
    
    return tab, run_info


def sample_mvs_ilocs(tab, params, sample_params):
    '''Displays interesting rows following inference'''
    # Select rows to display based on result
    num_rows_to_display = sample_params.get('num_rows_to_display', 30)
    num_per_missing_val_to_display = sample_params.get('sample_params', 4)
    
    row_idxs = []
    for col, mvs in params['mvs_dict']['columns'].items():
        for mv in mvs:
            sel = (tab[col] == mv['val']).diff().fillna(True)
            sel.index = range(len(sel))
            row_idxs.extend(list(sel[sel].index)[:num_per_missing_val_to_display])
            
    if not row_idxs:
        row_idxs = range(num_rows_to_display)

    return row_idxs    


if __name__ == '__main__':
    
    file_paths = ['../../data/test_dedupe/participants.csv', 
                  '../../data/test/etablissements/bce_data_norm.csv',
                  'local_test_data/source.csv']
    file_path = file_paths[-1] # Path to file to test
    
    nrows = 100000 # How many lines of the file to read for inference
    encoding = 'utf-8' # Input encoding
    tab = pd.read_csv(file_path, nrows=nrows, encoding=encoding, dtype='unicode')
    
    # Frequent missing value expressions
    PROBABLE_MISSING_VALUES = ['nan', 'none', 'na', '\\n', ' ', 'non renseigne', 'nr', 'no value', 'null', 'missing value']
    ALWAYS_MISSING_VALUES = ['']
    num_top_values = 15 # Number of most frequent values to look at
         
    # Guess expressions for missing values
    params = {'probable_mvs': PROBABLE_MISSING_VALUES,
              'always_mvs': ALWAYS_MISSING_VALUES, 
              'num_top_values': num_top_values}
    infered_params = infer_mvs(tab, params)
    print(infered_params)
    
    # Replace in original table
    params = {'mvs_dict': infered_params['mvs_dict'],
              'thresh': 0.6}
    tab = replace_mvs(tab, params)
