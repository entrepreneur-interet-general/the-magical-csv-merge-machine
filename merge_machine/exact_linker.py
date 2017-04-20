#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:57:59 2017

@author: leo
"""
import numpy as np
import pandas as pd

from dedupe_linker import pre_process

def create_new_column(tab, cols, sep=' '):
    """Merges columns in list cols to """
    # Create new_column name
    new_col_name = '_pre_processed_' + '__'.join(cols)
    while new_col_name in tab.columns:
        new_col_name += '__new'
    tab.loc[:, new_col_name] = tab.loc[:, cols[0]].fillna('')
    for col in cols[1:]:
        tab.loc[:, new_col_name] += sep + tab.loc[:, col].fillna('')
    tab.loc[:, new_col_name] = tab.loc[:, new_col_name].str.strip(sep)
    sel = tab.loc[:, new_col_name] == ''
    tab.loc[sel, new_col_name] = np.nan

    # Pre-process column
    sel = tab[new_col_name].notnull()
    tab.loc[sel, new_col_name] = tab.loc[sel, new_col_name].apply(pre_process)    

    return tab, new_col_name
        

def exact_linker(paths, params):
    """
        paths={'ref': 'path_to_reference',
               'source': 'path_to_source'}
        params={ 
                'columns': [{'source': ['col_1', 'col_2'], 'ref': 'col_3'}],
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
            if isinstance(pair[file_role], str):
                pair[file_role] = [pair[file_role]]
                
            if len(pair[file_role]) == 0:
                raise Exception('No column specified for {0} in pair of  \
                                matching columns'.format(file_role))
            else:
                tabs[file_role], new_col_name = create_new_column(tabs[file_role], 
                                    pair[file_role], sep=params.get('sep', ' '))
                cols_to_match_on[file_role].append(new_col_name)
                
    # Merge
    tabs['source'] = tabs['source'].merge(tabs['ref'], how='left', 
            left_on=cols_to_match_on['source'], right_on=cols_to_match_on['ref'],
            indicator=True)
    
    # Remove pre-processed columns
    #    for col in cols_to_match_on['source'] +  cols_to_match_on['ref']:
    #        for new_col in [col, col + '_x', col + '_y']:

    
    return tabs['source']

if __name__ == '__main__':
    paths = {
            'source': 'local_test_data/source.csv', 
             'ref': 'local_test_data/ref.csv'
             }
    params = {'columns': [{'source': 'lycees_sources',
                           'ref': ['denomination_principale_uai', 'patronyme_uai']},
                          {'source': 'commune', 
                           'ref': 'localite_acheminement_uai'}
                         ],
              'sep': ' '           
              }
    tab = exact_linker(paths, params)

    print(tab._merge.value_counts())