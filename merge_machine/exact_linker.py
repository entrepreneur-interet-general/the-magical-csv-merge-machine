#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:57:59 2017

@author: leo
"""
import numpy as np
import pandas as pd

def create_new_column(tab, cols, sep=' '):
    """Merges columns in list cols to """
    # Create new_column name
    new_col_name = '__'.join(cols)
    while new_col_name in tab.columns:
        new_col_name += '__new'
    tab.loc[:, new_col_name] = tab.loc[:, cols[0]].fillna('')
    for col in cols[1:]:
        tab.loc[:, new_col_name] += (sep + tab.loc[:, col].fillna('')).str.strip(sep)
    sel = tab.loc[:, new_col_name] == ''
    tab.loc[sel, new_col_name] = np.nan
    return tab, new_col_name
        

def exact_linker(paths, params):
    """
        paths={'ref': 'path_to_reference',
               'source': 'path_to_source'}
        params={ 
                'columns': [{'source': ['col_1', 'col_2'], 'ref': ['col_3']}],
                'sep': ' '
                }
    """
    # Load source and referential
    tabs = dict()
    tabs['source'] = pd.read_csv(paths['source'], encoding='utf-8')
    tabs['ref'] = pd.read_csv(paths['ref'], encoding='utf-8')
    
    # Create new columns to match on if necessary (columns concatenation)
    cols_to_match_on = {'source': [], 'ref': []}
    for pair in params['columns']:
        for file_role in ['source', 'ref']:
            if len(pair['file_role']) == 0:
                raise Exception('No column specified for {0} in pair of  \
                                matching columns'.format(file_role))
            elif len(pair[file_role]) == 1:
                cols_to_match_on[file_role].append(pair[file_role][0])
            else:
                tabs[file_role], new_col_name = create_new_column(tabs[file_role], 
                                    pair[file_role], sep=params.get('sep', ' '))
                cols_to_match_on[file_role].append(new_col_name)
                
    # Merge
    tabs['source'] = tabs['source'].merge(tabs['ref'], how='left', 
            left_on=cols_to_match_on['source'], right_on=cols_to_match_on['ref'])
    
    return tabs['source']
                
    