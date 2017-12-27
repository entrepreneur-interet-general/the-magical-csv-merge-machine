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
    
    Parameters
    ----------
    tab: `pandas.DataFrame`
        Result of merge through `es_linker`
    params: dict
        Additional information to check performances if both the source and the
        contain a column that can be used as a joining key that was NOT USED
        by es_linker. `params` should contain:
            
            col_matches: dict like {"source": col_source, "ref": col_ref}
                A pair of columns that can be used a joining key:
            lower: bool (defaults to False)
                Whether or not the values of the joining should be lowercased
                before joining
                
    Returns
    -------
    metrics: dict
        Information about the results of linking with `es_linker`. Fields are:
            
            perc_match_thresh: float
                The percentage of rows that have a match and a score above 
                the confidence threshold.
            num_match_thresh: int
                The number of rows that have a match and a score above 
                the confidence threshold.    
        
            perc_match: float
                The percentage of rows that have a possible match (any score).
            num_match: float
                The number of rows that have a possible match (any score).
            
            num_verif_samples: int
                The number of rows used for precision evaluation (0 if no 
                information is passed to `params`).
                
            precision: float between 0 and 1 (if `params` is passed)
                The estimated precision based on the number of matching values
                for the columns specified in `col_matches`.                
    '''
    
    if '__CONFIDENCE' not in tab.columns:
        raise Exception('Column __CONFIDENCE could not be found. Are you sure \
                        this is the result of a es_linker merge?')
    
    col_matches = params.get('col_matches', {})
    thresh = 1
    
    metrics = dict()

    tab.loc[:, '__CONFIDENCE'] = tab.__CONFIDENCE.astype(float)

    # Compute metrics
    metrics['perc_match'] = tab.__CONFIDENCE.notnull().mean() * 100
    metrics['num_match'] = int(tab.__CONFIDENCE.notnull().sum())
    
    # 
    metrics['perc_match_thresh'] = (tab.__CONFIDENCE >= thresh).mean() * 100
    metrics['num_match_thresh'] = int((tab.__CONFIDENCE >= thresh).sum())
    
    metrics['num_verif_samples'] = 0
    
    # Evaluate precision
    if col_matches:
        sel = tab[col_matches['source']].notnull() \
                & tab[col_matches['ref'] + '__REF'].notnull() \
                & (tab.__CONFIDENCE >= thresh)
                
        metrics['num_verif_samples'] = int(sel.sum())
        if sel.any():
            if params.get('lower', False):
                metrics['precision'] = (tab.loc[sel, col_matches['source']] \
                       == tab.loc[sel, col_matches['ref'] + '__REF']).mean()
            else:
                metrics['precision'] = (tab.loc[sel, col_matches['source']].str.lower() \
                       == tab.loc[sel, col_matches['ref'] + '__REF'].str.lower()).mean()
            metrics['perc_precision'] = metrics['precision'] * 100.
    else:
        metrics['num_verif_samples'] = 0
        
    return metrics
            