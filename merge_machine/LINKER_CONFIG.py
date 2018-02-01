#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:12:39 2018

@author: leo
"""

LANG = "french" # french, english, standard

# Compound analyzers 
c_a = {0: {'{0}', 'n_grams', 'integers', 'city', 'country'}, # General establishment / address
       1: {'{0}', 'n_grams', 'city'}, # city
       2: {'{0}', 'n_grams', 'country'}, # country
       3: {'{0}', 'n_grams', 'integers'}, # Non geographical text
       4: {'{0}', 'n_grams'}, # Non geographical, non numerical text
       5: {'n_grams', 'integers'}, # Code / ID / phone
       6: {}, # Exact match
       }

# Replace by language
c_a = {key: {string.format(LANG) for string in values} for key, values, in c_a.items()}

analyzers_to_use = \
    {'Académie': 0,
     'Adresse': 0,
     'Année': 5,
     'Article': 3,
     'Code INSEE': 5,
     'Code Postal': 5,
     'Commune': 1,
     'Corps et Grades': 4,
     'DOI': 5,
     'Date': 4,
     'Département': 3,
     'Email': 6,
     'Entité MESR': 0,
     'Entité agro': 3,
     'Entité biomédicale': 3,
     'Entreprise': 0,
     'Etablissement': 0,
     "Etablissement d'Enseignement Supérieur": 0,
     'Etablissement des premier et second degrés': 0,
     'ID': 5,
     'ID organisation': 5,
     'ID personne': 5,
     'ID publication': 5,
     'ISSN': 5,
     'Institution': 0,
     'Institution de recherche': 0,
     'Intitulé GRID': 0,
     'Mois': 5,
     'NIF': 5,
     'NIR': 5,
     'Nom': 4,
     'Nom de personne': 4,
     'Numéro National de Structure': 5,
     'Numéro UMR': 5,
     'Pays': 2,
     'Phyto': 3,
     'Prénom': 4,
     'Publication': 3,
     'Région': 0,
     'SIREN': 5,
     'SIRET': 5,
     'Spécialité médicale': 4,
     'Structure de recherche': 0,
     'Texte': 3,
     'Titre': 4,
     'Téléphone': 5,
     'UAI': 5,
     'URL': 5,
     'Voie': 5}

# The analzers to use to index each data type
DEFAULT_ANALYZERS_TYPE = {key: c_a[val] for key, val in analyzers_to_use.items()}

# Default analyzer (for non-matching columns)
DEFAULT_ANALYZER = 'case_insensitive_keyword'

# Default analyzers (for columns that should match)
DEFAULT_CUSTOM_ANALYZERS = {'case_insensitive_keyword', 'integers', 'n_grams', 'city', 'country'}
DEFAULT_STOCK_ANALYZERS = {'french'}
DEFAULT_ANALYZERS = DEFAULT_CUSTOM_ANALYZERS | DEFAULT_STOCK_ANALYZERS















"""
   "Etablissement": 0
    "Structure de recherche"
    "UAI"
    "Nom"
    "Numéro UMR"
    "Titre"
    "Corps et Grades"
    "Phyto"
    "SIRET"
    "ID publication"
    "Nom d'essai clinique"
    "Entreprise"
    "Département"
    "Code INSEE"
    "Partenaire de recherche"
    "Spécialité médicale"
    "Type structuré"
    "TVA"
    "Entité Géo"
    "Publication"
    "Entité MESR"
    "Institution de recherche"
    "NIF"
    "Numéro National de Structure"
    "Date"
    "Intitulé GRID"
    "Académie"
    "Article"
    "Texte"
    "Institution"
    "ID organisation"
    "Code Postal"
    "Région"
    "Voie"
    "DOI"
    "NIR"
    "ID"
    "Nom de personne"
    "SIREN"
    "Collaborateur d'essai clinique"
    "Téléphone"
    "Commune"
    "Mois"
    "Entité biomédicale"
    "Pays"
    "URL"
    "Education Nationale"
    "ISSN"
    "Prénom"
    "Entité agro"
    "Adresse"
    "Etablissement d'Enseignement Supérieur"
    "Email"
    "ID personne"
    "Année": 
    "Etablissement des premier et second degrés": 

"""