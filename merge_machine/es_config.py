#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:27:30 2017

@author: m75380
"""
import copy
import os

#curdir = os.path.dirname(os.path.realpath(__file__))
#os.chdir(curdir)

#city_keep_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_keep.txt')
#city_syn_file_path = os.path.join(curdir, 'resource', 'es_linker', 'es_city_synonyms.txt')
city_keep_file_path = 'es_city_keep.txt'
city_syn_file_path = 'es_city_synonyms.txt'

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
    },
    
    "my_city_keep" : {
        "type" : "keep",
        "keep_words_case": True, # Lower the words
        # "keep_words" : ["one", "two", "three"]
        "keep_words_path" : city_keep_file_path
    },
    "my_city_synonym" : {
        "type" : "synonym", 
        "expand": False,    
        "ignore_case": True,
        # "synonyms" : ["paris, lutece => paname"],
        "synonyms_path" : city_syn_file_path,
        "tokenizer" : "whitespace"  # TODO: whitespace? 
    },
    "my_length": {
        "type" : "length",
        "min": 4
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
    },
    'city': {
        "tokenizer": "whitespace", # TODO: problem with spaces in words
        "filter": ["my_city_keep", "my_city_synonym", "my_length"] # TODO: shingle ?
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
    Creates the dict to pass to index creation based on the columns_to_index
    
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

gen_index_settings = lambda columns_to_index: _gen_index_settings(index_settings_template, columns_to_index)