#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:54:21 2017

@author: leo
"""

def link_results_analyzer(tab, params={}):
    '''
    Takes a merged table, and a pair of columns that constitute a certain match. 
    Returns statistics on the number of good matches found etc...
    
    INPUT:
        - tab: pandas dataframe result of merge through dedupe_linker
        - params:
            col_matches: pair of columns that constitute certain matches:
                ex: {"source": "unique_id", "ref": "UNIQUE_ID"}
            lower: lowercase before matching
    
    '''
    
    if '__CONFIDENCE' not in tab.columns:
        raise Exception('Column __CONFIDENCE could not be found. Are you sure \
                        this is the result of a dedupe merge?')
    
    col_matches = params.get('col_matches', {})
    
    metrics = dict()

    # Compute metrics
    metrics['perc_match'] = tab.__CONFIDENCE.notnull().mean() * 100
    metrics['num_match'] = int(tab.__CONFIDENCE.notnull().sum())
    
    metrics['num_verif_samples'] = 0
    if col_matches:
        sel = tab[col_matches['source']].notnull() \
                & tab[col_matches['ref']].notnull() \
                & tab.__CONFIDENCE.notnull()
                
        metrics['num_verif_samples'] = int(sel.sum())
        if sel.any():
            if params.get('lower', False):
                metrics['precision'] = (tab.loc[sel, col_matches['source']] \
                       == tab.loc[sel, col_matches['ref']]).mean()
            else:
                metrics['precision'] = (tab.loc[sel, col_matches['source']].str.lower() \
                       == tab.loc[sel, col_matches['ref']].str.lower()).mean()
            metrics['perc_precision'] = metrics['precision'] * 100.
    else:
        metrics['num_verif_samples'] = 0
        
    return metrics
            