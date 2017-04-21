#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:57:59 2017

@author: leo
"""
import numpy as np
import pandas as pd

from dedupe_linker import pd_pre_process


def enrich_sirene(tab):
    '''Splits column L6_DECLAREE to create L6_DECLAREE.code_commune and L6_DECLAREE.localite'''
    tab.loc[:, 'L6_DECLAREE.code_commune'] = np.nan
    tab.loc[:, 'L6_DECLAREE.localite'] = np.nan
    
    sel = tab.L6_DECLAREE.str.slice(0, 5).str.isdigit().fillna(False)
    tab.loc[sel, 'L6_DECLAREE.code_commune'] = tab.L6_DECLAREE.str.slice(0, 5)
    tab.loc[sel, 'L6_DECLAREE.localite'] = tab.L6_DECLAREE.str.slice(5).str.strip()
    return tab


def create_new_column(tab, cols, sep=' '):
    """Merges columns in list cols to """
    
    if isinstance(cols, str):
        cols = [cols]
        
    if len(cols) == 0:
        raise Exception('Missing value in column pairs')
    else:    
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
        print('Pre-processing ', new_col_name)
        tab.loc[sel, new_col_name] = pd_pre_process(tab.loc[sel, new_col_name], 
                                                       remove_punctuation=True)
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
    CHUNKSIZE = 500000
    # Load source and referential
    print('Loading source')
    source = pd.read_csv(paths['source'], encoding='utf-8', sep=';', dtype=str)
    source.loc[:, '__OG_INDEX'] = source.index
    print('Loading ref')
    ref_chunks = pd.read_csv(paths['ref'], encoding='utf-8', sep=',', dtype=str, 
                             iterator=True, chunksize=CHUNKSIZE)

    print('Loading done')
    
    
    # source: Create new columns to match on if necessary (columns concatenation)
    cols_to_match_on = {'source': [], 'ref': []}
    for pair in params['columns']:
        source, new_col_name = create_new_column(source, pair['source'], 
                                                 sep=params.get('sep', ' '))
        cols_to_match_on['source'].append(new_col_name)
    
    
    for i, ref_chunk in enumerate(ref_chunks):
        ref_chunk = enrich_sirene(ref_chunk)
        
        print('At row ', i*CHUNKSIZE)
        for pair in params['columns']:
            ref_chunk, new_col_name = create_new_column(ref_chunk, pair['ref'], 
                                                     sep=params.get('sep', ' '))
            if i == 0:
                cols_to_match_on['ref'].append(new_col_name)        
        
        # Merge
        source_chunk = source.merge(ref_chunk, how='inner', 
                left_on=cols_to_match_on['source'], right_on=cols_to_match_on['ref'],
                indicator=True)
        
        
        if i == 0:
            new_source = source_chunk
        else:
            new_source = new_source.append(source_chunk)
            print(source.shape)
            print(new_source.shape)
            
            
    for col in source.columns:
        if (col in new_source.columns) and (col != '__OG_INDEX'):
            source.drop(col, axis=1, inplace=True)
    
    source = source.merge(new_source, on='__OG_INDEX', how='left')
    source.drop('__OG_INDEX', axis=1, inplace=True)
    
    # Remove pre-processed columns
    #    for col in cols_to_match_on['source'] +  cols_to_match_on['ref']:
    #        for new_col in [col, col + '_x', col + '_y']:

    
    return source

if __name__ == '__main__':
#    paths = {
#            'source': 'local_test_data/source.csv', 
#             'ref': 'local_test_data/ref.csv'
#             }
#    params = {'columns': [{'source': 'lycees_sources',
#                           'ref': ['denomination_principale_uai', 'patronyme_uai']},
#                          {'source': 'commune', 
#                           'ref': 'localite_acheminement_uai'}
#                         ],
#              'sep': ' '           
#              }



    paths = {
            'source': 'local_test_data/controles_sanitaires/export_alimconfiance.csv',
            'ref': 'local_test_data/sirene/petit_sirene.csv'
            }


#     ['APP_Libelle_activite_etablissement', 'APP_Libelle_etablissement',
#       'Adresse_activite', 'Code_postal', 'Localite', 'Libelle_commune',
#       'Date_inspection', 'Synthese_eval_sanit', 'Agrement', 'geores',
#       'ods_adresse', 'filtre', 'Date_extraction', 'Numero_inspection',
#       'SIRET']
    
    params = {'columns': [{'source': 'ods_adresse',
                           'ref': 'L4_DECLAREE'},
                          {'source': 'Localite', 
                           'ref': 'L6_DECLAREE.localite'},
                          {'source': 'APP_Libelle_etablissement', 
                           'ref': 'ENSEIGNE_OU_NOMEN_LONG'}
                         ],
              'sep': ' '           
              }

        
    
    tab = exact_linker(paths, params)

    print(tab._merge.value_counts())