#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380
"""
import copy

tokenizers = {
    "integers": {
        "type": "pattern",
        "preserve_original": 0,
        "pattern": '(\\d+)',
        'group': 1
    },
    "n_grams": {
        "type": "ngram",
        "min_gram": 3,
        "max_gram": 3,
        "token_chars": [
            "letter",
            "digit"
        ]
    }
}

filters = {
    "my_edgeNGram": {
        "type": "edgeNGram",
        "min_gram": 3,
        "max_gram": 30
    }
}

analyzers = {
    "integers": {
        'tokenizer': 'integers'
    },
    "n_grams": {
        'tokenizer': 'n_grams'
    },
    "end_n_grams": {
        'tokenizer': 'keyword',
        "filter": ["reverse", "my_edgeNGram", "reverse"]
    }
}


index_settings_template = {
    "settings": {
        "analysis": {
            "tokenizer": tokenizers,
            "filter": filters,
            "analyzer": analyzers
        }
    },

    "mappings": {
        "structure": {}
    }
}

def _gen_index_settings(index_settings_template, columns_to_index):
    '''
    NB: the default analyzer is keyword
    '''
    index_settings = copy.deepcopy(index_settings_template)
    
    field_mappings = {
        key: {
            'analyzer': 'keyword',
            'type': 'string',
            'fields': {
                analyzer: {
                    'type': 'string',
                    'analyzer': analyzer
                }
                for analyzer in values
            }
        }
        for key,
        values in columns_to_index.items() if values
    }
                
    field_mappings.update({
        key: {
            'analyzer': 'keyword',
            'type': 'string'
        }
        for key,
        values in columns_to_index.items() if not values
    })
                
    index_settings['mappings']['structure']['properties'] = field_mappings
    
    return index_settings

gen_index_settings = lambda x: _gen_index_settings(index_settings_template, x)