#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:44:35 2017

@author: m75380
"""

from merge_machine import es_match

from es_connection import es

def es_linker(source, params):
    source = es_match.es_linker(es, source, params)
    modified = source.copy() # TODO: is this good?
    modified.loc[:, :] = True
    return source, modified