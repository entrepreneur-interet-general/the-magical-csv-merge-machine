#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:18:28 2017

@author: leo
"""

from es_match import es_linker
from results_analyzer import link_results_analyzer
from missing_values import infer_mvs, replace_mvs, sample_mvs_ilocs
from preprocess_fields_v3 import infer_types, normalize_values, sample_types_ilocs
# from restrict_reference import infer_restriction, perform_restriction


NORMALIZE_MODULE_ORDER_log = ['INIT', 'make_mini', 'add_selected_columns', 
                              'infer_mvs', 'replace_mvs', 'infer_types',
                              'recode_types', 'concat_with_init']

NORMALIZE_MODULE_ORDER = ['INIT', 'make_mini', 'replace_mvs', 'recode_types', 
                          'concat_with_init']

LINK_MODULE_ORDER_log = ['INIT', 'add_selected_columns', 'es_train', 'upload_es_train', 
                         'es_linker', 'link_results_analyzer']
# NB: INIT in link is add_projects

# Old order using dedupe
#LINK_MODULE_ORDER_log = ['add_selected_columns', 'load_labeller', 'train', 
#                         'upload_train', 'dedupe_linker', 'infer_restriction', 
#                         'perform_restriction', 'link_results_analyzer']

LINK_MODULE_ORDER = [] # TODO

NORMALIZE_MODULES = {
        'transform':{
                    'INIT': {
                                'desc': 'Initial upload (cannot be called)'                            
                            },
                    'make_mini': {
                                'desc': 'Make __MINI__ version of selected file',
                                'write_to': 'INIT',
                            },
                    'replace_mvs': {
                                'func': replace_mvs,
                                'desc': replace_mvs.__doc__,
                                'use_in_full_run': True
                            },
                    'recode_types': {
                               'func':  normalize_values,
                               'desc': normalize_values.__doc__,
                               'use_in_full_run': True
                           },
                    'concat_with_init': {
                               'desc': 'Merge intial and transformed files (cannot be called)',
                               'use_in_full_run': True
                           }
                    },
        'infer':{
                'infer_mvs': {
                                'func': infer_mvs,
                                'write_to': 'replace_mvs',
                                'desc': infer_mvs.__doc__
                            },
                'infer_types': {
                               'func': infer_types,
                               'write_to': 'recode_types',
                               'desc': infer_types.__doc__
                           }
                },
        'sample': {
                'standard': {
                        'desc': 'Standard N-line sampler'
                        },                
                'sample_mvs': {
                        'func': sample_mvs_ilocs,
                        'desc': sample_mvs_ilocs.__doc__
                        },
                'sample_types': {
                        'func': sample_types_ilocs,
                        'desc': sample_types_ilocs.__doc__
                        }
                }
        }
       

LINK_MODULES = {
        'transform': {
                    'es_linker': {
                               'func': es_linker,
                               'desc': es_linker.__doc__,
                               'use_in_full_run': True
                            }
                    },
        'infer':{
                'link_results_analyzer': {
                                'func': link_results_analyzer,
                                'write_to': 'results_analysis',
                                'desc': link_results_analyzer.__doc__
                            },
#                'infer_restriction': {
#                                'func': infer_restriction,
#                                'write_to': 'restriction',
#                                'desc': link_results_analyzer.__doc__                        
#                            }
                },
        'link': {
    #                'dedupe_linker': {
    #                                'func': dedupe_linker,
    #                                'desc': dedupe_linker.__doc__
    #                            },
                },
        'sample': {
                'standard': {
                        'desc': 'Standard N-line sampler'
                        }
                }
        }
         
# Sanity check
for module_from_loop in NORMALIZE_MODULE_ORDER:
    assert (module_from_loop in NORMALIZE_MODULES['transform'])
    
for module_from_loop in LINK_MODULE_ORDER:
    assert (module_from_loop in LINK_MODULES['transform'])