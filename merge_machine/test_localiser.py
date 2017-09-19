#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:58:29 2017

@author: m75380

https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html
https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-keep-words-tokenfilter.html
https://data.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000%40public/export/?disjunctive.country

Valid match if no token on one of both sides

"""
from collections import defaultdict
import json
import os

import requests

file_path = os.path.join('resource', 'es_linker', 'geonames-all-cities-with-a-population-1000.json')

with open(file_path) as f:
    res = json.load(f)

no_country_count = 0
no_alternate_count = 0

cities = set()
countries = set()
cities_to_countries = defaultdict(set)
cities_to_cities = defaultdict(set)
city_hashes = defaultdict(set)
for i, row in enumerate(res):
    if i%10000 == 0:
        print('Did {0}/{1}'.format(i, len(res)))
        
    my_row = row['fields']
    name = my_row['name']
    if 'alternate_names' in my_row:
        alternates = my_row['alternate_names'].split(',')
    else:
        alternates = []
        no_alternate_count += 1
        
    if 'country' in my_row:
        country = my_row['country']
        
        cities.update([name] + alternates)
        countries.add(country)
        for city in [name] + alternates:
            cities_to_countries[city].add(country)
            cities_to_cities[city].update([name] + alternates)
            city_hashes[city].add(name)
    else:
        no_country_count += 1

# Generate synonym file for ES
file_path_syn = os.path.join('resource', 'es_linker', 'es_city_synonyms.txt')
file_path_keep = os.path.join('resource', 'es_linker', 'es_city_keep.txt')
with open(file_path_syn, 'w') as w_syn, \
     open(file_path_keep, 'w') as w_keep:
    for key, values in city_hashes.items():
        # sea biscuit, sea biscit => seabiscuit
        string = key + ' => ' + ', '.join(values) + '\n'
        w_syn.write(string)
        w_keep.write(key + '\n')
        
        
assert False

url = 'https://raw.githubusercontent.com/David-Haim/CountriesToCitiesJSON/master/countriesToCities.json'

try:
    country_to_cities
except:
    country_to_cities = json.loads(requests.get(url).content.decode('utf-8'))

all_cities = set()
city_to_countries = defaultdict(list)
for country, cities in country_to_cities.items():
    
    if country != '':
        all_cities.update(cities)
        for city in set(cities):
            city_to_countries[city].append(country)
    
    
