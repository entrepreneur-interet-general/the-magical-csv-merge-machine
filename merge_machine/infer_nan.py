#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:11:07 2017

@author: leo

Take table data and infer values that represent missing data to replace by 
standard missing data.

Examples of missing data that can be found:
    - "XXX"
    - "NO_ADDR"
    - "999999"
    - "NONE"
    - "-"
    
    
    
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
    

def mv_from_len_diff(top_values, score=1):
    """Check if all values have the same length except one"""
    
    # Compute lengths of values
    lengths = pd.Series([len(x) for x in top_values.index], index=top_values.index)
    # Check if all values have the same length except one:
    if lengths.nunique() == 2 & (len(top_values) >= 4):
        
        if lengths.value_counts().iloc[-1] == 1:
            mv_value = lengths.value_counts().index[-1]
            return [(mv_value, score, 'diff')]
    return []

def mv_from_len_ratio(top_values, score=0.2):
    """Check if value is much shorter than others"""
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
    """Check if value is the only not digit"""
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
    for col_1, top_values_1 in all_top_values.iteritems():
        for col_2, top_values_2 in all_top_values.iteritems():
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
        temp = [(val, len(cols)) for val, cols in popular_values.iteritems()]
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
    for col, tuples in col_mvs.iteritems():
        for (val, score, origin) in tuples:
            if val not in val_mvs:
                val_mvs[val] = [col]
            else:
                val_mvs[val].append(col)
                
    return [(val, score, 'common_values') for val, cols in val_mvs.iteritems() if (len(cols)>=2)]
    

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
            new_list_of_possible_mvs_tmp[val] = coef
        else:
            new_list_of_possible_mvs_tmp[val] += coef
                                    
    # NB: Taken care of in mv_from_usual_forms
    #        # If the value is a known form of mv, increase probability 
    #        if val.lower() in [x.lower() for x in probable_mvs]:
    #            new_list_of_possible_mvs_tmp[val] += 0.5

    # Reformat output like input
    new_list_of_possible_mvs = []
    for val, coef in new_list_of_possible_mvs_tmp.iteritems():
        new_list_of_possible_mvs.append((val, coef, 'unknown'))
    
    return new_list_of_possible_mvs


def infer_mvs(tab, probable_mvs=[u'nan'], always_mvs=[u''], num_top_values=10):
    """Run mv inference processes for each column and for the entire table"""
    # Compute most frequent values per column
    all_top_values = compute_all_top_values(tab, num_top_values)
    
    col_mvs = dict()
    # Look at each column and infer mv
    for col, top_values in all_top_values.iteritems():
        col_mvs[col] = []
        col_mvs[col].extend(mv_from_len_diff(top_values))
        col_mvs[col].extend(mv_from_len_ratio(top_values))
        col_mvs[col].extend(mv_from_not_digit(top_values))
        col_mvs[col].extend(mv_from_punctuation(top_values))
        col_mvs[col].extend(mv_from_usual_forms(top_values, probable_mvs))
        col_mvs[col].extend(mv_from_usual_forms(top_values, always_mvs, 10**3))
        col_mvs[col].extend(mv_from_letter_repetition(top_values))
        col_mvs[col] = correct_score(col_mvs[col], probable_mvs)
        col_mvs[col].sort(key=lambda x: x[1], reverse=True)
        
    
    common_mvs = mv_from_common_values_2(col_mvs)
    return {'columns': {key:val for key, val in col_mvs.iteritems() if val}, 
            'all': common_mvs}

def replace_mvs(tab, mvs_dict, thresh=0.6):
    """
    Replace the values that should be mvs by actual np.nan. Values in 'all' 
    will be replaced in the entire table whereas values in 'columns' will only
    be replaced in the specified columns.
    
    INPUT:
        tab: pandas DataFrame to modify
        infered_mvs: dict indicating mv values with scores. For example:
                {
                    'all': [],
                    'columns': {u'dech': [(u'-', 2.0, 'unknown')],
                                u'distance': [(u'-', 1, 'unknown')]}
                }
        thresh: minimum score to remove mvs
    
    OUTPUT:
        tab: same table with values replaced by np.nan
    
    """
    assert infered_mvs.keys() == ['all', 'columns']
    
    for (val, score, origin) in infered_mvs['all']:
        if score >= thresh:
            tab.replace(val, np.nan, inplace=True)
    
    for col, mv_values in infered_mvs['columns'].iteritems():
        for (val, score, origin) in mv_values:
            if score >= thresh:
                tab[col].replace(val, np.nan, inplace=True)
    return tab



if __name__ == '__main__':
    
    file_paths = ['data/test_dedupe/participants.csv']
    file_path = file_paths[-1] # Path to file to test
    
    nrows = 100000 # How many lines of the file to read for inference
    encoding = 'utf-8' # Input encoding
    tab = pd.read_csv(file_path, nrows=nrows, encoding=encoding, dtype='unicode')
    
    # Frequent missing value expressions
    PROBABLE_MISSING_VALUES = [u'nan', u'none', u'na', u'\\n', u' ', 'non renseigne', 
                     'no value', 'null', 'missing value']
    ALWAYS_MISSING_VALUES = [u'']
    num_top_values = 15 # Number of most frequent values to look at
         
    # Guess expressions for missing values
    infered_mvs = infer_mvs(tab, PROBABLE_MISSING_VALUES, ALWAYS_MISSING_VALUES, num_top_values)
    print(infered_mvs)
    
    # Replace in original table
    tab = replace_mvs(tab, infered_mvs, thresh=0.6)
