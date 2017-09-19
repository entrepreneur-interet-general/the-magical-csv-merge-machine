#!/usr/bin/env python3
# coding=utf-8

# Standard modules
import csv, itertools, re, logging, optparse, time, sys, math, os, unidecode
from functools import partial, reduce, lru_cache
from collections import defaultdict, Counter, Iterable
from operator import itemgetter, add
from fuzzywuzzy import fuzz
import stdnum.fr.nir, stdnum.fr.nif, stdnum.fr.siren, stdnum.fr.siret, stdnum.fr.tva
import pandas as pd
import numpy as np

# Parsing/normalization packages
import custom_name_parsing
import urllib.request, urllib.parse, json # For BAN address API on data.gouv.fr
from dateparser import DateDataParser
import phonenumbers

from CONFIG import RESOURCE_PATH

lastTime = 0
timingInfo = Counter()
countInfo = Counter()
MICROS_PER_SEC = 1000000

# Ratio of unhandled, non-fatal errors after which a given TypeMatcher will be dropped
MAX_ERROR_RATE = 20
# Number of unhandled, fatal errors after which a given TypeMatcher will be dropped
MAX_FATAL_COUNT = 5
# If set to True, all errors will be treated as fatal
FAIL_FAST_MODE = True

def snapshot_timing(end):
	global lastTime
	if lastTime > 0 and lastTime + MICROS_PER_SEC > end: return
	if lastTime > 0:
		logging.debug('Timing info')
		for k, v in timingInfo.most_common(50): logging.debug(k.rjust(50), str(v))
		logging.debug('')
	lastTime = end

TIMING_BY_INSTANCE = False
def timed(original_func):
	def wrapper(*args, **kwargs):
		start = time.clock()
		result = original_func(*args, **kwargs)
		end = time.clock()
		key = original_func.__name__ if len(args) < 1 else (args[0] if TIMING_BY_INSTANCE else args[0].__class__.__name__) + '.' + original_func.__name__
		t = int((end - start) * MICROS_PER_SEC)
		timingInfo[key] += t
		countInfo[key] += 1
		snapshot_timing(end)
		return result
	return wrapper

def chngrams(string="", n=3, top=None, threshold=0, exclude=[], **kwargs):
	""" Returns a dictionary of (character n-gram, count)-`s.
		N-grams in the exclude list are not counted.
		N-grams whose count falls below (or equals) the given threshold are excluded.
		N-grams that are not in the given top most counted are excluded.
	"""
	# An optional dict-parameter can be used to specify a subclass of dict,
	# e.g., count(words, dict=readonlydict) as used in Document.
	count = defaultdict(int)
	if n > 0:
		for i in xrange(len(string)-n+1):
			w = string[i:i+n]
			if w not in exclude:
				count[w] += 1
	if threshold > 0:
		count = dict((k, v) for k, v in count.items() if v > threshold)
	if top is not None:
		count = dict(heapq.nsmallest(top, count.items(), key=lambda kv: (-kv[1], kv[0])))
	return kwargs.get("dict", dict)(count)

class ApproximateLookup:
	def __init__(self, maxHits = 10):
		self.words = set()
		self.index = {}
		self.maxHits = maxHits
	def add(self, item):
		self.words.add(item)
	def remove(self, item):
		self.words.discard(item)
	def indexkeys(self, w):
		L = len(w)
		res = set([w])
		for i in range(L):
			for j in range(i,L):
				res.add(w[:i]+w[i+1:j]+w[j+1:])
		return res
	def makeindex(self):
		self.index.clear()
		for w in self.words:
			for key in self.indexkeys(w):
				if key in self.index: self.index[key].add(w)
				else: self.index[key] = set([w])
	def search(self, query):
		res = {0:[], 1:[], 2:[]}
		candidate = set()
		for key in self.indexkeys(query):
			if key in self.index: candidate.update(self.index[key])
		for word in candidate:
			dist = edit_dist(word, query)
			if dist < 3 and len(res) < self.maxHits: res[dist].append(word)
		return res

# Fast Levenshtein distance implementation

def edit_dist(s,t):
	matrix = {}
	for i in range(len(s)+1):
		matrix[(i, 0)] = i
	for j in range(len(t)+1):
		matrix[(0, j)] = j

	for j in range(1,len(t)+1):
		for i in range(1,len(s)+1):
			if s[i-1] == t[j-1]:
				matrix[(i, j)] = matrix[(i-1, j-1)]
			else:
				matrix[(i, j)] = min([matrix[(i-1, j)] +1, matrix[(i, j-1)]+1, matrix[(i-1, j-1)] +1])

	return matrix[(i,j)]

# Misc utilities

def flatten_list(l): return '' if l is None else l if isinstance(l, str) else '; '.join([flatten_list(v) for v in uniq(l)])

def uniq(sequence):
	''' Maintains the original sequence order. '''
	unique = []
	[unique.append(item) for item in sequence if item not in unique]
	return unique

# Natural language toolbox and lexical stuff: FRENCH ONLY!!!

STOP_WORDS = [
	# Prepositions (excepted "avec" and "sans" which are semantically meaningful)
	"a", "au", "aux", "de", "des", "du", "par", "pour", "sur", "chez", "dans", "sous", "vers",
	# Articles
	"le", "la", "les", "l", "c", "ce", "ca",
	 # Conjonctions of coordination
	"mais", "et", "ou", "donc", "or", "ni", "car"
]

def is_stop_word(word): return word in STOP_WORDS

def is_valid_phrase(tokens): return len(tokens) > 0 and not all(len(t) < 2 and t.isdigit() for t in tokens)

def stripped(s): return s.strip(" -_.,'?!").strip('"').strip()

def is_valid_token(token):
	token = stripped(token)
	if token.isspace() or not token: return False
	if token.isdigit(): return False # Be careful this does not get called when doing regex or template matching!
	if len(token) <= 2 and not (token.isalpha() and token.isupper()): return False
	return not is_stop_word(token)

def is_valid_value(v):
	''' Validates a single value (sufficient non-empty data and such things) '''
	stripped = stripped(v)
	return len(stripped) > 0 and stripped not in ['null', 'NA', 'N/A']

def is_acronym_token(token):
	return re.match("[A-Z][0-9]*$", token) or re.match("[A-Z0-9]+$", token)

MIN_ACRO_SIZE = 3
MAX_ACRO_SIZE = 6

def lower_or_not(token, keepAcronyms, keepInitialized = False):
	''' Set keepAcronyms to true in order to improve precision (e.g. a CAT scan will not be matched by a kitty). '''
	if keepAcronyms and len(token) >= MIN_ACRO_SIZE and len(token) <= MAX_ACRO_SIZE and is_acronym_token(token):
		return token
	if keepInitialized:
		m = re.search("([A-Z][0-9]+)[^'a-zA-Z].*", token)
		if m:
			toKeep = m.group(0)
			return toKeep + lower_or_not(token[len(toKeep):], keepAcronyms, keepInitialized)
	return token.lower()

def to_ASCII(phrase): return unidecode.unidecode(phrase)

def rejoin(v): return to_ASCII(v)

def case_phrase(p, keepAcronyms = False):
	ps = pre_split(p) if p is not None else ''
	return to_ASCII(lower_or_not(ps.strip(), keepAcronyms))

def replace_by_space(str, *patterns): return reduce(lambda s, p: re.sub(p, ' ', s), patterns, str)

def dehyphenate_token(token):
	result = token[:]
	i = result.find('-')
	while i >= 0 and i < len(result) - 1:
		left = result[0:i]
		right = result[i+1:]
		if left.isdigit() or right.isdigit(): break
		result = left + right
		i = result.find('-')
	return result.strip()

def pre_split(v):
	s = ' ' + v.strip() + ' '
	s = replace_by_space(s, '[\{\}\[\](),\.\"\';:!?&\^\/\*-]')
	return re.sub('([^\d\'])-([^\d])', '\1 \2', s)

# def fold_for_changes(s, normalizeCase = False):
#     ''' Override this if we want to skip minor changes (case, hyphenation, etc.) '''
#     return split_and_case(s) if normalizeCase else s

def normalize_and_validate_tokens(phrase,
	keepAcronyms = False, tokenValidator = is_valid_token, phraseValidator = is_valid_phrase, stopWords = None):
	''' Returns a list of normalized, valid tokens for the input phrase (an empty list
		if no valid tokens were found) '''
	if phrase:
		tokens = str.split(case_phrase(phrase, keepAcronyms))
		validTokens = []
		for token in tokens:
			if tokenValidator(token) and (stopWords is None or token not in stopWords): validTokens.append(token)
		if phraseValidator(validTokens): return validTokens
	return []

def normalize_and_validate_phrase(value,
	keepAcronyms = False, tokenValidator = is_valid_token, phraseValidator = is_valid_phrase, stopWords = None):
	''' Returns a string that joins normalized, valid tokens for the input phrase
		(None if no valid tokens were found) '''
	tokens = normalize_and_validate_tokens(value, keepAcronyms = keepAcronyms, tokenValidator = tokenValidator, 
		phraseValidator = phraseValidator, stopWords = stopWords)
	return ' '.join(tokens) if len(tokens) > 0 else None

def validated_lexicon(lexicon, tokenize = False):
	return set(filter(lambda v: v is not None, [normalize_or_not(s, tokenize) for s in lexicon]))

def add_to_lexicon_map(lexical_map, s, tokenize, stopWords):
	k = normalize_and_validate_phrase(s, stopWords = stopWords) if tokenize else case_phrase(s)
	if k is None: return
	lexical_map[k].append(s)

def cased_multi_map(mm):
	return dict([(case_phrase(k), set([case_phrase(v) for v in vs])) for k, vs in mm.items()])

def validated_lexical_map(lexicon, tokenize = False, stopWords = None, synMap = None):
	''' Returns a dictionary from normalized string to list of original strings. '''
	syn_map0 = None if synMap is None else cased_multi_map(synMap)
	lexical_map = defaultdict(list)
	for s in lexicon:
		if syn_map0 is not None:
			for (main_var, alt_vars) in synMap.items():
				s = re.sub(r"\b(%s)\b" % '|'.join(alt_vars), main_var, s, re.I)
		add_to_lexicon_map(lexical_map, s, tokenize, stopWords)
	return lexical_map

# Loading CSV and raw (one entry per line) text files

def file_row_iter(fileName, sep, path = RESOURCE_PATH):
	filePath = fileName if path is None else os.path.join(path, fileName)
	with open(filePath, mode = 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter = sep, quotechar='"')
		for row in reader:
			try:
				yield list(map(stripped, row))
			except UnicodeDecodeError as ude:
				logging.error('Unicode error while parsing "%s"', row)

def file_column_to_list(fileName, c, sep = '\t', includeInvalid = True):
	return [r[c] for r in file_row_iter(fileName, sep) if len(r) > c and (includeInvalid or is_valid_value(r[c]))]

FRENCH_LEXICON = file_column_to_list('most_common_tokens_fr', 0, '|')

def file_to_list(fileName, path = RESOURCE_PATH):
	filePath = fileName if path is None else os.path.join(path, fileName)
	with open(filePath, mode = 'r') as f:
		return [stripped(line) for line in f]

# TODO improve following function :
# - to accept a varargs with several filenames
# - to automatically handle multiple languages by checking suffixes ("_en", "_fr") without having to enumerate them by hand
def file_to_set(fileName): return set(file_to_list(fileName))

def file_to_variant_map(fileName, sep = '|', includeSelf = False, tokenize = False):
	''' The input format is pipe-separated, column 1 is the main variant, column 2 an alternative variant.

		Returns a reverse index, namely a map from original alternative variant to original main variant

		Parameters:
		includeSelf if True, then the main variant will be included in the list of alternative variants
			(so as to enable partial matching simultaneously).
	'''
	otherToMain = dict()
	mainToOther = defaultdict(list)
	for row in file_row_iter(fileName, sep):
		if len(row) < 2: continue
		r = list([normalize_or_not(e, tokenize = tokenize)  for e in row])
		main, alts = r[0], set(r[1:])
		alts.discard(main)
		for alt in alts: 
			if alt not in otherToMain:
				otherToMain[alt] = main
			elif main not in otherToMain and otherToMain[alt] != main:
				otherToMain[main] = otherToMain[alt]
		mainToOther[main].extend(list(alts))
	l = list(otherToMain.items())
	if includeSelf: l = list([(main, main) for main in mainToOther.keys()]) + l
	return dict(l)

# Template extraction and matching

def char_templates(c): return 'L' if c.isalpha() else 'D' if c.isdigit() else 'A'

def char_template(c): return 'D' if c.isdigit() else ('L' if c.isalpha() else '?')

MPL = (5, 16) # Min and max pattern length
def token_templates(token, simple = True):
	''' Parameters:
		simple if True, then this method only builds full patterns (i.e. no strict prefixes) and also just considers
		L and D (not A) '''
	if simple:
		if len(token) >= MPL[0] and len(token) <= MPL[1]:
			candidate = list([char_template(token[n]) for n in range(len(token))])
			if 'D' in candidate: yield '{}$'.format(''.join(candidate))
	else:
		l = list()
		for n in range(len(token)):
			l.append(''.join(char_templates(token[n])))
			if n < MPL[0] or 'D' not in l: continue
			for p in itertools.product(l):
				yield p
				if n == len(token) - 1: yield '{}$'.format(p)

# Ngram extraction and matching

def ngram_iter(v, n, bounds = False):
	if len(v) < n: return iter(())
	return chngrams(' {} '.format(v) if bounds else v, n).items()

# Special datatype categories
C_OTHERS = 'Autres types'

# Special fields and field prefixes
# F_COMPOSITE_REMAINDER = u'+++'
F_ORIGINAL_PATTERN = u'Original %s'
F_ACRONYMS = u'Acronyme' # For on-the-fly acronym detection (i.e. when acronym and expanded form occur in the same context)
F_VARIANTS = u'Variante' # For other forms of synonyms, including acronyms from pre-collected acro/expansion pair files

# Generic, MESR-domain and other field names

F_PERSON = u'Nom de personne'
F_FIRST = u'Prénom'
F_LAST = u'Nom'
F_TITLE = u'Titre'
F_JOURNAL = u'Titre de revue'

F_EMAIL = u'Email'
F_URL = u'URL'
F_INSEE = u'Code INSEE'
F_YEAR = u'Année'
F_MONTH = u'Mois'
F_DATE = u'Date'
F_PHONE = u'Téléphone'

F_GEO = u'Entité Géo'
F_ADDRESS = u'Adresse'
F_ZIP = u'Code Postal'
F_COUNTRY = u'Pays'
F_CITY = u'Commune'
F_STREET = u'Voie'
F_HOUSENUMBER = u'Numéro de voie'
F_DPT = u'Département'
F_REGION = u'Région'

F_STRUCTURED_TYPE = u'Type structuré'
F_TEXT = u'Texte'

F_ENGLISH = u'Anglais'
F_FRENCH = u'Français'

F_ID = u'ID'
F_ORG_ID = u'ID organisation'
F_PERSON_ID = u'ID personne'
F_ENTREPRISE = u'Entreprise' # Nom ou raison sociale de l'entreprise
F_SIREN = u'SIREN'
F_SIRET = u'SIRET'
F_NIF = u'NIF' # Numéro d'Immatriculation Fiscale
F_NIR = u'NIR' # French personal identification number
F_TVA = u'TVA'
F_GRID_LABEL = u'Intitulé GRID'

F_PUBLI = u'Publication'
F_ARTICLE = u'Article'
F_ABSTRACT = u'Résumé'
F_ISSN = u'ISSN'
F_ARTICLE_CONTENT = u'Contenu d\'article'
F_PUBLI_ID = u'ID publication'
F_DOI = u'DOI'

F_MESR = u'Entité MESR'
F_NNS = u'Numéro National de Structure'
F_UAI = u'UAI'
F_UMR = u'Numéro UMR'
F_RD_STRUCT = u'Structure de recherche'
F_RD_PARTNER = u'Partenaire de recherche'
F_CLINICALTRIAL_COLLAB = u'Collaborateur d\'essai clinique'
F_RD = u'Institution de recherche'
F_ETAB = u'Etablissement'
F_EDUC_NAT = u'Education Nationale'
F_ACADEMIE = u'Académie'
F_ETAB_NOTSUP = u'Etablissement des premier et second degrés'
F_ETAB_ENSSUP = u'Etablissement d\'Enseignement Supérieur'
F_APB_MENTION = u'Mention APB'
F_RD_DOMAIN = u'Domaine de Recherche'

# A very high-level institution, comprising
# 1. (higher) education entities
# 2. R&D organizations
# 3. entreprises/corporations
F_INSTITUTION = u'Institution'

F_CLINICALTRIAL_NAME = u'Nom d\'essai clinique'
F_MEDICAL_SPEC = u'Spécialité médicale'
F_BIOMEDICAL = u'Entité biomédicale'
F_PHYTO = u'Phyto'
F_AGRO = u'Entité agro'
F_RAISON_SOCIALE = u'Raison sociale'

# This mapping only covers supertype-subtype relationships
SUBTYPING_RELS = defaultdict(set)
# This mapping only covers composition relationships
COMPOSITION_RELS = defaultdict(set)
# This mapping covers both subtyping and composition relationships
PARENT_CHILD_RELS = defaultdict(set)

TYPE_TAGS = {
	F_PERSON: [F_LAST, F_PERSON],
	F_FIRST: [F_PERSON],
	F_LAST: [u'Patronyme', F_PERSON],
	F_EMAIL: [F_ADDRESS, u'Courriel'],
	F_URL: [F_ADDRESS],
	F_INSEE: [F_ID, u'Code', u'Numéro'],
	F_NIF: [F_ID], 
	F_NIR: [F_ID], 
	F_TVA: [F_ID], 
	F_GRID_LABEL: [F_ID, 'GRID', 'Recherche', 'Index'],
	F_YEAR: [F_DATE],
	F_MONTH: [F_DATE],
	F_PHONE: [u'Numéro'],
	F_ADDRESS: [F_GEO],
	F_ZIP: [u'CP', u'Code', F_ADDRESS, F_GEO],
	F_COUNTRY: [F_ADDRESS, F_GEO],
	F_CITY: [u'Ville', F_ADDRESS, F_GEO],
	F_STREET: [u'Rue', F_ADDRESS, F_GEO],
	F_DPT: [F_ADDRESS, F_GEO],
	F_REGION: [F_ADDRESS, F_GEO],
	F_ID: [u'Identifiant', u'Code', u'Numéro'],
	F_ORG_ID: [u'Organisation', u'Structure', u'Identifiant', u'Code', u'Numéro'],
	F_PERSON_ID: [u'Personne', u'Individu', u'Identifiant', u'Code', u'Numéro'],
	F_ENTREPRISE: [u'Société', u'Organisation'],
	F_SIREN: [u'Identifiant', u'Numéro'],
	F_SIRET: [u'Identifiant', u'Numéro'],
	F_PUBLI_ID: [F_PUBLI, u'Identifiant', u'Numéro'],
	F_DOI: [u'Identifiant', u'Numéro'],
	F_UAI: [u'Identifiant', u'Numéro'],
	F_UMR: [u'Identifiant', u'Numéro', u'Recherche'],
	F_RD_STRUCT: [u'Organisation', u'Structure', u'Recherche'],
	F_RD_PARTNER: [u'Organisation', u'Structure', u'Recherche'],
	F_RD: [u'Recherche'],
	F_EDUC_NAT: [u'Education', u'Enseignement'],
	F_ACADEMIE: [u'Enseignement'],
	F_ETAB: [u'Enseignement'],
	F_ETAB_NOTSUP: [u'Primaire', u'Secondaire', u'Lycée', u'Collège'],
	F_ETAB_ENSSUP: [u'Enseignement supérieur'],
	F_PHYTO: [u'Agro'],
	F_AGRO: [u'Agro'],
	F_MEDICAL_SPEC: [u'Médecine'],
	F_BIOMEDICAL: [u'Médecine']
}

# Stop words specific to certain data types

STOP_WORDS_CITY = ['commune', 'cedex', 'cdx']

# Base class for all type matchers

class TypeMatcher(object):
	def __init__(self, t):
		self.t = t
		self.diversion = set()
	def diversity(self):
		''' Specifies the min number of distinct reference values to qualify a column-wide match
			(it is essential to carefully override this constraint when the labels in question represent
			singleton instances, i.e. specific entities, as opposed to a qualified and/or controlled
			vocabulary). '''
		return 1
	def __str__(self):
		return '{}<{}>'.format(self.__class__.__name__, self.t)
	def register_full_match(self, c, t, s, hit = None):
		ms = cover_score(c.value, hit) if s is None else s
		outputFieldPrefix = None if self.t == t else self.t 
		c.register_full_match(t, outputFieldPrefix, ms, hit)
		self.update_diversity(hit)
	def register_partial_match(self, c, t, ms, hit, span):
		outputFieldPrefix = None if self.t == t else self.t 
		c.register_partial_match(t, outputFieldPrefix, ms, hit, span)
		self.update_diversity(hit)
	@timed
	def match_all_field_values(self, f):
		error_count = 0
		fatal_count = 0
		values_seen = 0
		for vc in f.cells:
			if values_seen >= 100 and (error_count + fatal_count) * 100 > values_seen * MAX_ERROR_RATE:
				logging.warning('{}: bailing out after {} total matching errors'.format(self, error_count + fatal_count))
				break
			elif fatal_count > MAX_FATAL_COUNT:
				logging.warning('{}: bailing out after {} fatal matching errors'.format(self, fatal_count))
				break
			values_seen += 1
			try :
				self.match(vc)
			# Handling non-fatal errors
			except ValueError as ve:
				logging.warning('{}: value error for "{}": {}'.format(self, vc.value, ve))
				if FAIL_FAST_MODE: fatal_count += 1
				else: error_count += 1
			except OverflowError as oe:
				logging.error('{} : overflow error (e.g. while parsing date) for "{}": {}'.format(self, vc.value, oe))
				if FAIL_FAST_MODE: fatal_count += 1
				else: error_count += 1
			# Handling fatal errors
			except RuntimeError as rte:
				logging.warning('{}: runtime error for "{}": {}'.format(self, vc.value, rte))
				fatal_count += 1
			except TypeError as te:
				logging.warning('{}: type or parsing error for "{}": {}'.format(self, vc.value, te))
				fatal_count += 1
			except UnicodeDecodeError as ude:
				logging.error('{} : unicode error while parsing input value "{}": {}'.format(self, vc.value, ude))
				fatal_count += 1
			except urllib.error.URLError as ue:
				logging.warning('{}: request rejected for "{}": {}'.format(self, vc.value, ue))
				fatal_count += 1
	def update_diversity(self, hit):
		self.diversion |= set(hit if isinstance(hit, list) else [hit])
	def check_diversity(self, cells):
		div = len(self.diversion)
		if div <= 0: return
		self.diversion.clear()
		if div < self.diversity():
			logging.info('Not enough diversity matches of type {} produced by {} ({})'.format(self.t, self, div))
		else:
			logging.info('Positing value type {} by {}'.format(self.t, self))
			for c in cells: c.posit_type(self.t)

class GridMatcher(TypeMatcher):
	@timed
	def __init__(self):
		super(GridMatcher, self).__init__(F_GRID_LABEL)
		gridding.init_gridding()
	def match_all_field_values(self, f):
		src_items_by_label = gridding.grid_label_set([vc.value for vc in f.cells])
		for vc in f.cells:
			item = src_items_by_label[vc.value]
			if 'grid' in item:
				self.register_full_match(vc, self.t, 100, item['label'])

MATCH_MODE_EXACT = 0
MATCH_MODE_CLOSE = 1

# stdnum-based matcher-normalizer class

class StdnumMatcher(TypeMatcher):
	@timed
	def __init__(self, t, validator, normalizer):
		super(StdnumMatcher, self).__init__(t)
		self.t = t
		self.validator = validator
		self.normalizer = normalizer
	@timed
	def match(self, c):
		nv = c.value
		if self.validator(nv):
			nv0 = self.normalizer(nv)
			self.register_full_match(c, self.t, 100, nv0)

# Regex-based matcher-normalizer class

class RegexMatcher(TypeMatcher):
	@timed
	def __init__(self, t, p, g = 0, ignoreCase = False, partial = False, validator = None, neg = False, wordBoundary = True):
		super(RegexMatcher, self).__init__(t)
		self.p = pattern_with_word_boundary(p)
		self.g = g
		self.flags = re.I if ignoreCase else 0
		self.partial = partial
		self.validator = validator
		self.neg = neg
		self.wordBoundary = wordBoundary
		logging.info('SET UP regex matcher for <%s> (length %d)', self.t, len(self.p))
	@timed
	def match(self, c):
		if self.partial:
			ms = re.findall(self.p, c.value, self.flags)
			if ms:
				if self.neg:
					c.negate_type(self.t)
					return
				else:
					for m in ms:
						if not isinstance(m, str):
							if len(m) > self.g: 
								m = m[self.g]
							else: 
								continue
						i1 = c.value.find(m)
						if i1 >= 0:
							if self.validator is None or self.validator(m):
								self.register_partial_match(c, self.t, 100, m, (i1, i1 + len(m)))
								return
						else:
							raise RuntimeError('{} could not find regex multi-match "{}" in original "{}"'.format(self, m, c.value))
		else:
			m = re.match(self.p, c.value, self.flags)
			if m:
				if self.neg:
					c.negate_type(self.t)
					return
				else:
					try:
						grp = m.group(self.g)
						if self.validator is None or self.validator(grp):
							if len(grp) == len(c.value):
								self.register_full_match(c, self.t, 100, grp)
							else:
								self.register_partial_match(c, self.t, 100, grp, (0, len(grp)))
							return
					except IndexError:
						raise RuntimeError('No group {} matched in regex {} for input "{}"'.format(self.g, self.p, c))
		raise ValueError('{} unmatched "{}"'.format(self, c.value))

def build_vocab_regex(vocab, partial):
	j = '|'.join(vocab)
	return '({}).*$'.format(j if partial else j)

class VocabMatcher(RegexMatcher):
	''' When matcher is not None, the matching will be dispatched to it, which is useful when normalization is far costlier 
		than type detection. '''
	@timed
	def __init__(self, t, vocab, ignoreCase = False, partial = False, validator = None, neg = False, matcher = None):
		super(VocabMatcher, self).__init__(t, build_vocab_regex(vocab, partial),
			g = 0, ignoreCase = ignoreCase, partial = partial, validator = validator, neg = neg)
		self.matcher = matcher
	@timed
	def match(self, c):
		if self.matcher is not None:
			logging.debug('%s normalizing "%s" from %s', self, c, self.matcher)
			self.matcher.match(c)
		elif self.matcher is None or self.t not in c.non_excluded_types():
			logging.debug('%s normalizing "%s" from vocab only', self, c)
			super(VocabMatcher, self).match(c)
		else:
			raise TypeError('{} matcher only found excluded type in "{}"'.format(self, c))

# Tokenization-based matcher-normalizer class

def tokenization_based_score(matchedSrcTokens, srcTokens, matchedRefPhrase, refPhrase,
	minSrcTokenRatio = 80, minSrcCharRatio = 70, minRefCharRatio = 60):
	srcTokenRatio = 100 * len(matchedSrcTokens) / len(srcTokens)
	if srcTokenRatio < minSrcTokenRatio: return 0
	srcCharRatio = 100 * sum([len(t) for t in matchedSrcTokens]) / sum([len(t) for t in srcTokens])
	if srcCharRatio < minSrcCharRatio: return 0
	matchedRefPhrase = ' '.join(matchedSrcTokens)
	refCharRatio = 100 * len(matchedRefPhrase) / len(refPhrase)
	return 0 if refCharRatio < minRefCharRatio else refCharRatio

def stop_words_as_normalized_list(stopWords): 
	return [] if stopWords is None else list([case_phrase(s, False) for s in stopWords])

DTC = 6 # Dangerous Token Count (becomes prohibitive to tokenize many source strings above this!)
class TokenizedMatcher(TypeMatcher):
	@timed
	def __init__(self, t, lexicon, maxTokens = 0, scorer = tokenization_based_score, distinctCount = 0, stopWords = None):
		super(TokenizedMatcher, self).__init__(t)
		currentMax = maxTokens
		self.scorer = scorer
		self.phrasesMap = validated_lexical_map(lexicon)
		self.tokenIdx = dict()
		self.distinctCount = distinctCount
		self.stopWords = stop_words_as_normalized_list(stopWords)
		for np in self.phrasesMap.keys():
			tokens = list([t for t in np.split(' ') if t not in self.stopWords])
			if len(tokens) < 1: continue
			if maxTokens < 1 and len(tokens) > currentMax:
				currentMax = len(tokens)
				if currentMax > DTC:
					logging.warning('Full tokenization of lexicon: encountered token of length {}, above DTC!'.format(currentMax))
			matchedRefPhrase = ' '.join(tokens[:currentMax])
			if matchedRefPhrase not in self.tokenIdx or len(self.tokenIdx[matchedRefPhrase]) < len(np):
				self.tokenIdx[matchedRefPhrase] = np
		self.maxTokens = currentMax
		logging.info('SET UP %d-token matcher (%s-defined length) for <%s> with lexicon of size %d, total variants %d',
			self.maxTokens, 'user' if maxTokens > 0 else 'data', self.t, len(self.phrasesMap), len(self.tokenIdx))
	def diversity(self):
		return self.distinctCount if self.distinctCount > 0 else math.log(len(self.phrasesMap), 1.5)
	@timed
	def match(self, c):
		tokens = normalize_and_validate_tokens(c.value, tokenValidator = lambda t: is_valid_token(t) and t not in self.stopWords)
		if tokens is not None:
			for k2 in range(self.maxTokens, 0, -1):
				for k1 in range(0, len(tokens) + 1 - k2):
					matchSrcTokens = tokens[k1:k1 + k2]
					matchRefPhrase = ' '.join(matchSrcTokens)
					if matchRefPhrase in self.tokenIdx:
						nm = self.tokenIdx[matchRefPhrase]
						score = self.scorer(matchSrcTokens, tokens, matchRefPhrase, nm)
						if nm not in self.phrasesMap:
							raise RuntimeError('Normalized phrase {} not found in phrases map'.format(nm))
						hit = self.phrasesMap[nm]
						v = case_phrase(c.value)
						# The next line joins on '' and not on ' ' because non-pure space chars might have been transformed
						# during tokenization (hyphens, punctuation, etc.)
						subStr = ''.join(matchSrcTokens)
						span = check_non_consecutive_subsequence(v, subStr)
						if span is None:
							logging.warning('%s could not find tokens "%s" in original "%s"', self, matchRefPhrase, v)
							span = (0, len(c.value))
						self.register_partial_match(c, self.t, score, hit, span)
						return
		raise ValueError('{} found no matching token sequence in "{}"'.format(self, c))

# Label-based matcher-normalizer class and its underlying FSS structure

def build_fast_sim_struct(terms):
	fss = ApproximateLookup()
	for term in terms: fss.add(term)
	fss.makeindex()
	return fss

def fast_sim_score(r, l = 1024):
	''' Return a pair (list of matched substrings, score) '''
	if len(r[0]) > 0: return (r[0], 100)
	elif len(r[1]) < 1 and len(r[2]) < 1: return ([], 0)
	elif l > 8: return (r[1], 50) if len(r[1]) > 0 else (r[2], 20)
	elif l > 4: return (r[1], 20) if len(r[1]) > 0 else (r[2], 10)
	else: return (r[1], 5) if len(r[1]) > 0 else ([], 0)

def normalize_or_not(v, tokenize = False, stopWords = None): 
	return normalize_and_validate_phrase(v, stopWords = stopWords) if tokenize else case_phrase(v)

class LabelMatcher(TypeMatcher):
	@timed
	def __init__(self, t, lexicon, mm, stopWords = None, synMap = None):
		''' Parameters:
			diversity specifies the min number of distinct reference values to qualify a column-wide match
				(it is essential to have this constraint when the labels in question represent singleton instances,
				i.e. specific entities, as opposed to a qualified and/or controlled vocabulary. '''
		super(LabelMatcher, self).__init__(t)
		self.mm = mm
		self.stopWords = stop_words_as_normalized_list(stopWords)
		self.synMap = synMap
		self.tokenize = synMap is not None
		# dictionary from normalized string to list of original strings
		labelsMap = validated_lexical_map(lexicon, tokenize = self.tokenize, stopWords = self.stopWords, synMap = synMap) 
		self.labelsMap = labelsMap
		if mm == MATCH_MODE_EXACT:
			logging.info('SET UP exact label matcher for <%s>: lexicon of size %d', self.t, len(labelsMap))
		elif mm == MATCH_MODE_CLOSE:
			self.fss = build_fast_sim_struct(labelsMap.keys())
			logging.info('SET UP close label matcher for <%s>: lexicon of size %d', self.t, len(labelsMap))
	def diversity(self):
		return math.log(len(self.labelsMap), 1.8)
	@timed
	def match(self, c):
		v = normalize_or_not(c.value, stopWords = self.stopWords, tokenize = self.tokenize)
		if not v:
			raise TypeError('{} found no non-trivial label value from "{}"'.format(self, c))
		if self.synMap is not None and v in self.synMap: # Check for synonyms
			v = self.synMap[v]
		if self.mm == MATCH_MODE_EXACT:
			if v in self.labelsMap:
				self.register_full_match(c, self.t, 100, self.labelsMap[v])
				return
		elif self.mm == MATCH_MODE_CLOSE:
			(matchedRefPhrases, score) = fast_sim_score(self.fss.search(v), len(v))
			if score > 0:
				for matchedRefPhrase in matchedRefPhrases:
					self.register_full_match(c, self.t, score, matchedRefPhrase)
					return
		raise ValueError('{} found no matching label in "{}"'.format(self, c))		

class HeaderMatcher(LabelMatcher):
	def __init__(self, t, lexicon):
		super(HeaderMatcher, self).__init__(t, lexicon, MATCH_MODE_CLOSE)

# Subtype matcher-normalizer class

class SubtypeMatcher(TypeMatcher):
	def __init__(self, t, subtypes):
		super(SubtypeMatcher, self).__init__(t)
		self.subtypes = set(subtypes)
		if len(subtypes) < 1: raise Error('Invalid subtype matcher setup')
		logging.info('SET UP subtype matcher for <%s> with subtypes: %s', self.t, ', '.join(subtypes))
		PARENT_CHILD_RELS[t] |= set(subtypes)
		SUBTYPING_RELS[t] |= set(subtypes)
	def match(self, c):
		sts = list(self.subtypes & c.non_excluded_types())
		if len(sts) < 1: return None
		ps = 0
		ms = None
		fs = 0
		tis = list()
		for st in sts:
			pss = c.matches(st, PARTIAL_MATCH)
			if len(pss) > 0:
				ps = max([ps] + [ti.ms for ti in pss])
				tis.extend(pss)
			fss = c.matches(st, FULL_MATCH)
			if len(fss) > 0:
				fs = max([fs] + [ti.ms for ti in fss])
				tis.extend(fss)
		if ps > 0 or fs > 0: c.register_cover_match(self.t, max(ps, fs), tis)

# Composite type matcher-normalizer class

class CompositeMatcher(TypeMatcher):
	def __init__(self, t, compTypes):
		super(CompositeMatcher, self).__init__(t)
		self.compTypes = compTypes
		if len(compTypes) < 1: raise RuntimeError('Invalid composite matcher setup')
		logging.info('SET UP composite matcher for <%s> with %d types', self.t, len(compTypes))
		PARENT_CHILD_RELS[t] |= set(compTypes)
		COMPOSITION_RELS[t] |= set(compTypes)
	@timed
	def match(self, c):
		sts = list(set(self.compTypes) & c.non_excluded_types())
		if len(sts) < 1: return None
		ps = 0
		ms = None
		fs = 0
		tis = list()
		for st in sts:
			pss = c.matches(st, PARTIAL_MATCH)
			if len(pss) > 0:
				ps += sum([ti.ms for ti in pss])
				tis.extend(pss)
			fss = c.matches(st, FULL_MATCH)
			if len(fss) > 0:
				fs += sum([ti.ms for ti in pss])
				tis.extend(fss)
		if ps > 0 or fs > 0: c.register_cover_match(self.t, max(ps, fs) / len(sts), tis)

class CompositeRegexMatcher(TypeMatcher):
	''' This class is useful when several children fields of a parent composite field
		can be captured as groups within the same regex.

		It assumes that all children types have been matched against, hence it
		only registers matches on the composite type itself. '''
	def __init__(self, t, p, tgs, ignoreCase = False, partial = False, validators = { }):
		''' Parameters:
			tgs a dictionary { type: group } to capture multiple types '''
		super(CompositeRegexMatcher, self).__init__(t)
		self.tgs = tgs
		self.p = pattern_with_word_boundary(p)
		self.flags = re.I if ignoreCase else 0
		self.partial = partial
		self.validators = validators
	def match(self, c):
		if self.partial:
			ms = re.findall(self.p, v, flags = self.flags)
			if not ms: return
			for m in ms:
				for (t, g) in self.tgs.items():
					try:
						grp = m.group(g)
						if t not in self.validators or self.validators[t](grp):
							self.register_partial_match(c, t, 100, grp, m.span(g))
					except IndexError:
						logging.error('No group %d matched in %s for input %s', g, self.p, c)
		else:
			m = re.match(self.p, c.value, flags = self.flags)
			if not m: return
			for (t, g) in self.tgs.items():
				try:
					grp = m.group(g)
					if t not in self.validators or self.validators[t](grp):
						if len(grp) == len(c.value):
							self.register_full_match(c, t, 100, grp)
						else:
							self.register_partial_match(c, t, 100, grp, (0, len(grp)))
				except IndexError:
					logging.error('No group %d matched in regex "%s" for input "%s"', g, self.p, c)

# Object model representing our inference process

# Drop inferred types falling below this column-wide threshold (in percent):
COLUMN_SCORE_THRESHOLD = 10
# For supertype relationships only (not composite-type!), switch from parent to child type when:
#   parent's score < this ratio * child's score
SUBTYPING_RATIO = 2
# SUPERTYPE_RATIO = 2 # Switch from parent to child type when: parent's score < this ratio * child's score
# COMPTYPE_RATIO = 2 # Switch from composite to component type when: composite's score < this ratio * child's score

def parse_fields_from_CSV(fileName, delimiter):
	''' Takes a CSV filepath and a delimiter as input, returns an instance of the Fields class. '''
	a = list(file_row_iter(fileName, delimiter, path = None))
	return Fields({ Cell(h, h): Field([Cell(a[i][k] if len(a[i]) > k else '', h) for i in range(1, len(a))]) for (k, h) in enumerate(a[0]) },
		len(a) - 1)

def parse_fields_from_Panda(df):
	''' Takes a DataFrame as input, returns an instance of the Fields class. '''
	return Fields({ Cell(h, h): Field([Cell(v, h) for v in c]) for (h, c) in df.items() },
		df.shape[0])

class Fields(object):
	def __init__(self, fields, entries):
		self.fields = fields # Mapping from header Cell object to value Field object
		self.entries = entries
		self.modifiedByColumn = { }
		self.outputFieldsByColumn = { }
	@timed
	def match_headers_and_values(self):
		logging.info('RUNNING all header matchers')
		for hm in header_matchers():
			for hc in self.fields.keys():
				logging.debug('RUNNING %s on %s header', hm, hc.value)
				try:
					hm.match(hc)
				except:
					pass # Ignore errors on header matching since no need to bail out (few values to check)
		logging.info('RUNNING all value matchers')
		for vm in value_matchers():
			if isinstance(vm, SubtypeMatcher): continue
			if isinstance(vm, CompositeMatcher): continue
			for (hc, f) in self.fields.items():
				logging.debug('RUNNING %s on %s values', vm, hc.value)
				vm.match_all_field_values(f)
				vm.check_diversity(f.cells)
	def match_by_types(self, trusted_types):
		for vm in value_matchers():
			if isinstance(vm, SubtypeMatcher): continue
			if isinstance(vm, CompositeMatcher): continue
			for (hc, f) in self.fields.items():
				if hc.value not in trusted_types:
					continue
				if vm.t != trusted_types[hc.value]:
					continue
				logging.debug('TRUSTING %s on %s values', vm, hc.value)
				vm.match_all_field_values(f)
	def likeliest_types(self, h, f, singleType = False):
		''' Returns None rather than an empty list to signify that not a single type has been inferred.

			Parameters:
			h the field header
			f the field itself
			singleType if True, then a unique choice per column will be made '''
		lht = h.likeliest_type()
		logging.info('Likeliest type for %s header: %s', h.value, lht)
		lvts = [lht] if lht is not None else None
		if singleType:
			lvt = f.likeliest_type()
			if lvt is not None:
				logging.info('Likeliest type for %s values: %s', h.value, lvt)
				lvts = [lvt]
		else:
			lvt = f.likeliest_types()
			if len(lvt) > 0:
				logging.info('Likeliest types for %s values: %s', h.value, ', '.join(lvt))
				lvts = lvt
		if lvts is None:
			logging.info('Could not infer type for %s values: %s', h.value, ', '.join(lvt))
		return lvts
	def process_values(self, outputFormat, singleType = False):
		''' Parameters:
			outputFormat either "md" for markdown output, or a separator string for CSV output '''
		self.match_headers_and_values()
		ofs = list()
		fts = dict()
		for (h, f) in self.fields.items():
			lvts = self.likeliest_types(h, f, singleType = True)
			if lvts is None: continue
			fts[h] = lvts
			hofs = uniq(list(itertools.chain.from_iterable([f.normalized_fields(h, lvt) for lvt in lvts])))
			logging.info('Output fields for %s: %s', h, hofs)
			ofs.extend(hofs)
		logging.info('Output fields for all: %s', ofs)
		# Compute new cell values
		b = list()
		for i in range(self.entries):
			l = list()
			for j in range(len(ofs)): l.append(None)
			b.append(l)
		for (h, lvts) in fts.items():
			for lvt in lvts:
				for i, nc in enumerate(f.normalized_values(h, lvt)):
					for (j, of) in enumerate(ofs):
						if of not in nc: continue
						ncs = nc[of] if isinstance(nc[of], list) else [nc[of]]
						b[i][j] = ncs if b[i][j] is None else b[i][j] + ncs
		if outputFormat == 'md':
			# Print timing info
			print('## Temps de traitement')
			print('')
			print('|Classe et méthode|Temps total (ms)|')
			print('|-|-|')
			if len(timingInfo) > 0:
				for k, v in timingInfo.most_common(20):
					print('|{}|{}|'.format(k, str(v)))
			# Print headers
			print('## Résultats de normalisation')
			print('')
			print('|{}|'.format('|'.join(ofs)))
			print('|{}|'.format('|'.join('-' * len(ofs))))
		else:
			print(outputFormat.join(ofs))
		for i in range(self.entries):
			ovs = list([flatten_list(b[i][j]) for j in range(len(ofs))])
			if outputFormat == 'md':
				print('|{}|'.format('|'.join(ovs)))
			else:
				print(outputFormat.join(ovs))
	# The following two methods do the same thing as the previous one, but with redundant operations
	# (splitting them is required in order to provide separate API calls prior to deduping)
	@timed
	def infer_types(self):
		''' Returns a dictionary mapping input field name to likeliest type.
			Fields for which no type has been inferred will be missing from the output dictionary.'''
		self.match_headers_and_values()
		types = dict()
		f2t = defaultdict(list)
		t2f = defaultdict(list)
		for (h, f) in self.fields.items():
			fieldName = h.value
			lht = h.likeliest_type()
			logging.info('Likeliest type for %s header: %s', fieldName, lht)
			if lht is not None: types[fieldName] = lht
			for (t, s) in f.scored_types().items():
				if s < COLUMN_SCORE_THRESHOLD: continue
				f2t[fieldName].append((t, s))
				t2f[t].append((fieldName, s))
		for (h, f) in self.fields.items():
			fieldName = h.value
			ts = sorted(f2t[fieldName], key = itemgetter(1), reverse = True)
			logging.info('Sorted types for {}: {}'. format(fieldName, '; '.join(['{} ({}) '.format(t, s) for (t, s) in ts])))
			for (t, s) in ts:
				betterField = any((p[1] > s and p[0] not in types) for p in t2f[t])
				betterChild = t in SUBTYPING_RELS and any((p[1] * SUBTYPING_RATIO > s and p[0] in SUBTYPING_RELS[t]) for p in ts)
				if not (betterField or betterChild):
					logging.info('Likeliest type for %s values: %s', fieldName, t)
					types[fieldName] = t
					break
			if fieldName not in types: logging.info('Could not infer type for %s values', fieldName)
		return types
	@timed
	def normalize_values(self, types):
		''' Generates (field name, list of field values) pairs for each output field. '''
		self.modifiedByColumn.clear()
		self.outputFieldsByColumn.clear()
		for (h, f) in self.fields.items():
			fieldName = h.value
			self.modifiedByColumn[fieldName] = [0] * self.entries
			if fieldName not in types:
				logging.warning('No values to normalize for {}'.format(fieldName))
				continue
			logging.info('Normalizing values for {}'.format(fieldName))
			lvt = types[fieldName]
			ofs = uniq(list(f.normalized_fields(h, lvt)))
			logging.info('Output fields for %s: %s', fieldName, ofs)
			nvs = list(f.normalized_values(h, lvt))
			for of in ofs:
				b = [None] * self.entries
				for i, nc in enumerate(nvs):
					if of not in nc or nc[of] is None: continue
					if b[i] is None:
						b[i] = nc[of]
					else:
						if isinstance(b[i], list):
							if isinstance(nc[of], list): 
								b[i].extend(nc[of])
							else: 
								b[i].append(nc[of])
						else:
							if isinstance(nc[of], list): 
								b[i] = [b[i]] + nc[of]
							else: 
								b[i] = [b[i], nc[of]]
					oldValue = f.cells[i].value
					newValues = list([s for s in (nc[of] if isinstance(nc[of], list) else [nc[of]])])
					isNewValue = len(newValues) > 0 and oldValue not in newValues
					if isNewValue: 
						self.modifiedByColumn[fieldName][i] += 1
				yield (of, b)
			self.outputFieldsByColumn[fieldName] = ofs
	def normalize_values_in_place(self, types):
		''' Generates (original field name, modified field values) pairs for each output field and each cell that is 
		 actually modified (even just changing a single character's case). '''
		for (h, f) in self.fields.items():
			fieldName = h.value
			if fieldName not in types:
				logging.warning('No values to normalize for {}'.format(fieldName))
				continue
			logging.info('Normalizing values for {}'.format(fieldName))
			lvt = types[fieldName]
			assert self.entries == len(f.cells)
			newCol = [''] * self.entries
			mod_count = 0
			for i, c in enumerate(f.cells):
				nvs = list(c.normalized_values_in_place(lvt))
				newCol[i] = ', '.join(nvs)
				if c.value != newCol[i]: 
					logging.debug('Field {} recoded: {} --> {}'.format(fieldName, c.value, newCol[i]))
					mod_count += 1
			logging.debug('Field {} recoded: {} / {} total'.format(fieldName, mod_count, len(f.cells)))
			yield (fieldName, newCol)

@lru_cache(maxsize = 1048576, typed = False)
def cached_normalized_values(c, t): return c.normalized_values(t)

class Field(object):
	def __init__(self, cells):
		# List of Cell objects
		self.cells = cells
	def scored_types(self):
		candidateTypes = reduce(set.union, [set(c.tis.keys()) for c in self.cells])
		# Map from type to a list of individual scores
		typeScores = { t: [0] * len(self.cells) for t in candidateTypes }
		for (i, c) in enumerate(self.cells):
			nets = c.non_excluded_types()
			for (t, tis) in c.tis.items():
				if t not in nets: continue
				s = max(ti.ms for ti in tis)
				if s > 0: typeScores[t][i] = s
		return { t: non_zero_ratio_score(scores) for (t, scores) in typeScores.items() }
	def likeliest_types(self):
		matchingTypes = self.scored_types()
		return sorted(matchingTypes.keys(), key = lambda t: matchingTypes[t], reverse = True) if len(matchingTypes) > 0 else []
	def likeliest_type(self):
		lts = self.likeliest_types()
		if len(lts) > 0:
			for (i, lt) in enumerate(lts):
				if lt not in PARENT_CHILD_RELS or len(PARENT_CHILD_RELS[lt] & set(lts[i + 1:])) < 1: return lt
		return None
	def normalized_fields(self, h, t):
		# return reduce(set.union, [set(cached_normalized_values(c, t).keys()) for c in self.cells], set([h.value]))
		return reduce(set.union, [set(c.normalized_values(t).keys()) for c in self.cells], set([h.value]))
	def normalized_values(self, h, t):
		''' Casts this field with header h into type t and returns its values as a list of augmented
			(field name, field value) dictionaries (not including the original field with its header). '''
		for c in self.cells:
			# Normalized/augmented fields
			nc = c.normalized_values(t)
			# Original field value
			nc[h.value] = c.value
			yield nc

PARTIAL_MATCH = 0
FULL_MATCH = 1
class TypeInference(object):
	def __init__(self, t, mm, ms, hit, i1, i2):
		self.t = t
		self.mm = mm # Match mode (partial or full, as an integer enum)
		self.ms = ms # Match score (in [0, 100])
		self.hit = hit # Matching string
		self.span = (i1, i2) if (0 <= i1 and i1 < i2) else None # Start and end index in the original string (used to compute coverage)
	def __repr__(self): return 'TI<{}>: {} <-- {}'.format(self.t, self.ms, self.hit)
	def __str__(self): return '<{}>'.format(self.t)

def cmp_hits(h1, h2):
	if h1.span and h2.span:
		c = h1.span[0] - h2.span[0] # Match beginning first
		if c != 0: return c
		d = h2.span[1] - h1.span[1] # Match end last
		if d != 0: return d
	return h2.ms - h1.ms # Higher score first

def checkSpan(hit, span):
	if not span or span[0] < 0 or span[1] <= span[0]:
		logging.warning('Invalid span for hit=%s: %s', hit, span)

def cover_score(s1, s2):
	return len(s1) * 100 / len(s2) if len(s1) <= len(s2) else cover_score(s2, s1)

CONVERT_NAN_TO_EMPTY = True

class Cell(object):
	def __init__(self, value, fieldName):
		# Original value, of type string
		self.value = value
		if CONVERT_NAN_TO_EMPTY and self.value is np.nan:
			self.value = ''
		self.f = fieldName
		# Map from type to a list of TypeInferences
		self.tis = defaultdict(list)
		# Negated types
		self.nts = set()
		# Posited (i.e. sufficiently diversified) types
		self.pts = set()
		# Mapping from normalized, augmented, or otherwise enriched field name to list of values for that field
		self.values = dict()
	def __str__(self): return '{}: {}'.format(self.f, self.value)
	def value_to_match(self):
		if self.value is not None:
			s = stripped(self.value)
			if s is not None: return s
		return None
	def negate_type(self, t):
		logging.debug('Negated type {} for "{}"'.format(t, self.value))
		self.nts.add(t)
	def posit_type(self, t):
		''' Does the opposite of negating this type: more precisely, it indicates that there is enough diversity
			across the entire value set, so that *if* any matcher for the type has enough recall, the field-wide
			match will be accepted. '''
		self.pts.add(t)
	def non_excluded_types(self):
		return set(self.tis.keys()) & self.pts - self.nts
	def matches(self, t, mm):
		return [] if t not in self.non_excluded_types() else filter(lambda ti: ti.mm == mm, self.tis[t])
	def normalized_type(self, t, outputFieldPrefix):
		return '++{}++'.format(self.f if outputFieldPrefix is None or outputFieldPrefix == self.f else outputFieldPrefix + '.' + self.f)
	def register_full_match(self, t, outputFieldPrefix, ms, hit = None):
		''' If normedIfHeaderField is True and the target type is the cell's parent type, then the output
			field should indicate that normalization has been done in order not to conflict with the
			original field. '''
		if ms <= 0: return
		t0 = self.normalized_type(t, outputFieldPrefix)
		logging.debug('FULL MATCH of type <%s> for %s (p=%d): %s', t, self, ms, self.value if hit is None else hit)
		self.tis[t].append(TypeInference(t0, FULL_MATCH, ms, self.value if hit is None else hit, 0, len(self.value)))
	def register_partial_match(self, t, outputFieldPrefix, ms, hit, span):
		# TODO accept span = None and fetch start/end indices on-the-fly
		if ms <= 0: return
		checkSpan(hit, span)
		t0 = self.normalized_type(t, outputFieldPrefix)
		logging.debug('PARTIAL MATCH of type <%s> for %s (p=%d): %s', t, self, ms, hit)
		self.tis[t].append(TypeInference(t0, PARTIAL_MATCH, ms, hit, span[0] if span else -1, span[1] if span else -1))
	def register_cover_match(self, t, ms, tis):
		if any(ti.mm == FULL_MATCH for ti in tis):
			self.register_full_match(t, False, ms)
		elif len(tis) > 0:
			stis = sorted(tis, cmp = hitsCmp)
			hit, k = stis[0].hit, stis[0].span[1]
			for ti in tis: checkSpan(hit, ti.span)
			for j in range(1, len(stis)):
				if k >= stis[j].span[0]:  continue
				hit += (' ' + stis[j].hit)
				k = stis[j].span[1]
			self.register_partial_match(t, False, ms, hit, (stis[0].span[0], stis[-1].span[1]))
	def likeliest_type(self):
		if len(self.tis) < 1: return None
		scores = { t: max([ti.ms for ti in tis]) for (t, tis) in self.tis.items() }
		return sorted(scores.keys(), key = lambda t: scores[t], reverse = True)[0]
	def normalized_values(self, t):
		res = defaultdict(set)
		nt = self.type_to_normalize(t)
		if nt is not None:
			for ti in self.tis[nt]:
				if isinstance(ti.hit, list): res[ti.t] |= set(ti.hit)
				else: res[ti.t].add(str(ti.hit))
		nvs = dict()
		for (k, v) in res.items():
			ucvs = unique_cell_values(v)
			for i, ucv in enumerate(filter(lambda ucv: ucv is not None and len(ucv) > 0, ucvs)):
				if i == 0: nvs[k] = ucv
				else: nvs['{}.{}'.format(k, i + 1)] = ucv
		return nvs
	def normalized_values_in_place(self, t):
		res = defaultdict(set)
		s = set()
		nt = self.type_to_normalize(t)
		if nt is not None:
			for ti in self.tis[nt]:
				if isinstance(ti.hit, list): res[ti.t] |= set(ti.hit)
				else: res[ti.t].add(str(ti.hit))
			for (k, v) in res.items():
				ucvs = unique_cell_values(v)
				s |= set([ucv for ucv in ucvs if ucv is not None and len(ucv) > 0])
		return list(s) if len(s) > 0 else [self.value]
	def type_to_normalize(self, t):
		if t in self.tis and t not in self.nts: 
			whole_score = self.max_type_score(t)
			if t in COMPOSITION_RELS:
				for ct in COMPOSITION_RELS[t]:
					if ct in self.tis and ct not in self.nts: 
						part_score = self.max_type_score(ct)
						if part_score > whole_score: return ct
			return t
		if t in SUBTYPING_RELS:
			for ct in SUBTYPING_RELS[t]:
				if ct in self.tis and ct not in self.nts: return ct
		return None
	def max_type_score(self, t):
		return 0 if t not in self.tis else max([ti.ms for ti in self.tis[t]])

def set_as_list_or_singleton(v):
	s = set(v)
	if len(s) < 1: return 0
	l = list(s)
	return l[0] if len(l) < 2 else l

def unique_cell_values(vs, maxListLength = 3):
	''' Returns a unique value per cell, or None as appropriate. '''
	def select_best_value(vs): 
		return None if len(vs) < 1 else sorted(vs, key = lambda v: len(v), reverse = True)[0]
	def merger_by_token_list(v1): 
		return ' '.join(v1)
	em = defaultdict(set) # an equivalence map
	for v in vs:
		if v is None or len(v) < 1: continue
		key = normalize_and_validate_phrase(v)
		if key is None: continue
		em[merger_by_token_list(v)].add(v)
	return list([select_best_value(vs) for vs in em.values()])[:maxListLength]

# TODO improve this custom date matcher if necessary by doing a first pass on all values at type inference time:
# - matcher 1 with languages = ['fr', 'en'],
#     settings = { 'DATE_ORDER': 'MDY', 'PREFER_LANGUAGE_DATE_ORDER': False } (the more likely one)
# - matcher 2 with languages = ['fr', 'en'],
#    settings = { 'DATE_ORDER': 'DMY', 'PREFER_LANGUAGE_DATE_ORDER': False }
# - then retain the majority matcher for the subsequent type inference pass (and the normalization step as well)
DDP = DateDataParser(languages = ['fr', 'en'], settings = { 'PREFER_LANGUAGE_DATE_ORDER': True })

class CustomDateMatcher(TypeMatcher):
	def __init__(self):
		super(CustomDateMatcher, self).__init__(F_DATE)
	@timed
	def match(self, c):
		# TODO check prioritization of ambiguous dates: the most complex case would be first encountering some FR date(s),
		# then an ambiguous date implicitly using US (not UK or AUS) locale (e.g. 03/04/2014 which should resolve to
		# March 4th and not April 3rd)...
		if c.value.isdigit():
			raise TypeError('{} cannot parse date from numeric value "{}"'.format(self, c))
		dd = DDP.get_date_data(c.value)
		do, dp = dd['date_obj'], dd['period']
		if do is None:
			raise TypeError('{} found no date field from "{}"'.format(self, c))
		y = do.year
		if y < 1870 or 2120 < y:
			raise ValueError('{} found no realistic date from "{}"'.format(self, c))
		ds = str(y)
		if dp == 'year':
			self.register_full_match(c, F_YEAR, 100, ds)
			return
		ds = "{:02}/{}".format(do.month, ds)
		if dp == 'month':
			self.register_full_match(c, F_MONTH, 100, ds)
		else:
			self.register_full_match(c, F_DATE, 100, "{:02}/{}".format(do.day, ds))

def score_phone_number(z): return 100 if phonenumbers.is_valid_number(z) else 75 if phonenumbers.is_possible_number(z) else 5

def normalize_phone_number(z): return phonenumbers.format_number(z, phonenumbers.PhoneNumberFormat.INTERNATIONAL)

class CustomTelephoneMatcher(TypeMatcher):
	def __init__(self, partial = False):
		super(CustomTelephoneMatcher, self).__init__(F_PHONE)
		self.partial = partial
	@timed
	def match(self, c):
		if partial:
			for match in phonenumbers.PhoneNumberMatcher(c.value, 'FR'):
				# original string is in match.raw_string
				self.register_partial_match(c, self.t, 100, normalize_phone_number(match.number), (match.start, match.end))
		else:
			z = phonenumbers.parse(c.value, 'FR')
			score = score_phone_number(z)
			if score > 0: self.register_full_match(c, self.t, score, normalize_phone_number(z))

# Various regexes for strict type matchers
PAT_EMAIL = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PAT_URL = "(https?|ftp)://[^\s/$.?#].[^\s]*"

# Person-name matcher-normalizer code

PRENOM_LEXICON = file_to_set('prenom')
PATRONYME_LEXICON = file_to_set('patronyme_fr')

PAT_FIRST_NAME = '(%s)' % '|'.join([p for p in PRENOM_LEXICON])
PAT_LAST_NAME = '([A-Z][A-Za-z]+\s?)+'
PAT_LAST_NAME_ALLCAPS = '([A-Z][A-Z]+\s?)+'
PAT_INITIAL = '([A-Z][\.\-\s]{1,3}){1,3}'

# Ignore case on these two
PAT_FIRST_LAST_NAME = '\s*%s\s+(%s)\s*' % (PAT_FIRST_NAME, PAT_LAST_NAME)
PAT_LAST_FIRST_NAME = '\s*(%s)\s+%s\s*' % (PAT_LAST_NAME, PAT_FIRST_NAME)

# Don't ignore case on those two
PAT_FIRSTINITIAL_LAST_NAME = '\s*(%s)\s+((%s)|(%s))\s*' % (PAT_INITIAL, PAT_LAST_NAME, PAT_LAST_NAME_ALLCAPS)
PAT_LAST_FIRSTINITIAL_NAME = '\s*((%s)|(%s))\s+(%s)\s*' % (PAT_LAST_NAME, PAT_LAST_NAME_ALLCAPS, PAT_INITIAL)

def pattern_with_word_boundary(p): return '\\b' + p + '\\b'

def regex_with_word_boundary(p, flags = 0): return re.compile(pattern_with_word_boundary(p), flags)

PERSON_NAME_EXTRACTION_PATS = [
	(regex_with_word_boundary(PAT_FIRST_NAME, re.IGNORECASE), 1, -1),
	(regex_with_word_boundary('%s\s+(%s)' % (PAT_FIRST_NAME, PAT_LAST_NAME), re.IGNORECASE), 1, 2),
	(regex_with_word_boundary('(%s)\s+%s' % (PAT_LAST_NAME, PAT_FIRST_NAME), re.IGNORECASE), 3, 1),
	(regex_with_word_boundary('(%s)\s+((%s)|(%s))' % (PAT_INITIAL, PAT_LAST_NAME, PAT_LAST_NAME_ALLCAPS)), 1, 2),
	(regex_with_word_boundary('((%s)|(%s))\s+(%s)' % (PAT_LAST_NAME, PAT_LAST_NAME_ALLCAPS, PAT_INITIAL)), 2, 1)
]

def validate_first_name(fst): return len(fst) > 1

def validate_last_name(lst): return len(lst) > 2

def validate_person_name(s):
	''' Validator for items of type: full person name '''
	for i, (r, firstGp, lastGp) in enumerate(PERSON_NAME_EXTRACTION_PATS):
		m = r.match(s)
		if m:
			logging.debug('Person name pattern #%d matched: %s', i + 1, '; '.join(m.groups()))
			fst = m.group(firstGp).strip()
			lst = (m.group(lastGp) if lastGp >= 0 else s.replace(m.group(firstGp), '')).strip()
			if validate_first_name(fst) and validate_last_name(lst):
				return { F_FIRST: fst, F_LAST: lst }
	return None

def validate_person_name_match(t):
	v = validate_person_name(t[0])
	return None if v is None else (v, t[1], t[2])

def singleton_list(s, itemValidator, stripChars):
	''' Parameters:
		itemValidator takes an input string and returns a dictionary
			{field name -> field value if the item is validated or else None}.

		Returns a list (of length one) whose element is a (field name, field value) dictionary. '''
	if itemValidator is None: return [s]
	try:
		d = itemValidator(s)
		return [] if d is None else [d]
	except:
		return []

DELIMITER_TOKENS_RE = re.compile(re.escape('et|and|&'), re.IGNORECASE)

def parse_person_name_list(s, itemValidator, i1 = 0, delimiters = ',;\t', stripChars = ' <>[](){}"\''):
	''' Parameters:
		itemValidator takes an input string and returns a list of mappings
			{ field name: field value if the item is validated or else None }.

		Returns a list of triples (item, startIndex, endIndex)
			where item is a (field name, field value) dictionary. '''
	s0 = DELIMITER_TOKENS_RE.sub(delimiters[0], s)
	for d in delimiters:
		(s1, s2, s3) = s0.partition(d)
		if len(s2) == 1:
			return singleton_list((s1, i1, i1 + len(s1)), validate_person_name_match, stripChars)
			+ parse_person_name_list(s3, validate_person_name_match, i1 + len(s1) + len(s2), delimiters = delimiters)
	return singleton_list((s0, i1, i1 + len(s0)), validate_person_name_match, stripChars)

##### START OF: specific, tailor-made parsing of person name lists

FR_FIRSTNAMES = map(str.lower, PRENOM_LEXICON)
FR_SURNAMES = map(str.lower, PATRONYME_LEXICON)
F_FIRSTORLAST = F_FIRST + '|' + F_LAST
PN_STRIP_CHARS = ' <>[](){}"\''
PN_DELIMITERS = ',;+/'
PN_TITLE_VARIANTS = {
	'M': ['mr', 'monsieur', 'mister', 'sir'], # Form "M." is handled separately
	'Mme': ['mme', 'mrs', 'madame', 'madam', 'ms', 'miss'],
	'Dr': ['dr', 'docteur', 'doctor'],
	'Pr': ['pr', 'professeur', 'prof', 'professor']
}

def person_name_singleton(s):
	d = extract_person_name(s)
	return [] if d is None else [d]

def extract_person_name(s):
	tokens = s.split()
	d = defaultdict(set)
	for token in tokens:
		if token[-1] in '-.':
			t = token.strip('-.')
			if len(t) > 1 or not t.isalpha() or t.upper() != t: continue
			if t == 'M': d[F_TITLE] = 'M'
			d[F_FIRST].add(t)
			continue
		t0 = token.strip('-.')
		if len(t0) < 2: continue
		t = t0.lower()
		title = None
		for (main, variants) in PN_TITLE_VARIANTS.items():
			if t0[1:].islower() and t in map(str.lower, variants):
				title = main
				break
		if title:
			d[F_TITLE].add(title)
			continue
		if t in FR_FIRSTNAMES:
			d[F_FIRST].add(t)
			d[F_FIRSTORLAST].add(t)
		elif t in FR_SURNAMES:
			d[F_LAST].add(t)
			d[F_FIRSTORLAST].add(t)
	if len(d[F_LAST]) < 1:
		if len(d[F_FIRSTORLAST]) > 0:
			d[F_LAST] = d[F_FIRSTORLAST] - d[F_FIRST]
			d[F_FIRST] = d[F_FIRST] - d[F_LAST]
			d[F_FIRSTORLAST] = d[F_LAST] - d[F_FIRST]
	if len(d[F_LAST]) < 1: return None
	if len(d[F_FIRST]) < 1:
		d[F_FIRST] = d[F_FIRSTORLAST] - d[F_LAST]
	del d[F_FIRSTORLAST]
	return d

##### END OF: specific, tailor-made parsing of person name lists

def parse_person_names(s):
	''' Returns None if validation failed, or else a list of triples (personName, startIndex, endIndex)
		where personName is a key/value dictionary.

		Patterns currently handled by this parser:
		<Last>
		<Last> <First>
		<First> <Last>
		<FirstInitial> <Last>
		<Last> <FirstInitial>
		LIST<Person> (with any kind of delimiter) '''
	return parse_person_name_list(s, validate_person_name)

class CustomPersonNameMatcher(TypeMatcher):
	def __init__(self):
		super(CustomPersonNameMatcher, self).__init__(F_PERSON)
	@timed
	def match(self, c):
		l = c.value
		parsed_names = custom_name_parsing.extractPersonNames(l)
		if len(parsed_names) < 1: return
		normalized_parsed_names = list([custom_name_parsing.joinPersonName(pn) for pn in parsed_names])
		self.register_full_match(c, self.t, 100, '; '.join(normalized_parsed_names))

# Phone number normalization

def normalize_phone_number(x):
	try:
		return phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
	except:
		return None

# Address matcher-normalizer class

LIBPOSTAL_MAPPING = [
	(F_STREET, ['house_number', 'road', 'suburb', 'city_district']),
	(F_ZIP, ['state', 'postcode']),
	(F_CITY, ['city']),
	(F_COUNTRY, ['country']) ]

BAN_MAPPING = [
	(F_HOUSENUMBER, 'housenumber'),
	(F_STREET, 'street'),
	(F_ZIP, 'postcode'),
	(F_CITY, 'city') ]


class CustomAddressMatcher(TypeMatcher):
	def __init__(self):
		super(CustomAddressMatcher, self).__init__(F_ADDRESS)
	@timed
	def match(self, c):
		if c.value.isdigit():
			logging.debug('Bailing out of %s for numeric value: %s', self, c)
			return
		parsed = parse_address(c.value)
		if not parsed: return
		ps = {key: value for (value, key) in parsed}
		v = c.value.lower()
		comps = list()
		for (f, lps) in LIBPOSTAL_MAPPING:
			comp = []
			for lp in lps:
				if lp not in ps: continue
				value = ps[lp]
				superStr = rejoin(v)
				subStr = rejoin(value)
				if len(superStr) != len(subStr):
					superStr = v
					subStr = value
				span = check_non_consecutive_subsequence(superStr, subStr)
				if span is None:
					logging.warning('%s could not find substring "%s" in original "%s"', self, subStr, superStr)
				else:
					self.register_partial_match(c, f, 100, subStr, span)
					comp.append(value)
			if len(comp) > 0: comps.append(' '.join(comp))
		if len(comps) > 0:
			self.register_full_match(c, self.t, None, ' '.join(comps))

COMMUNE_LEXICON = file_to_set('commune')
COUNTRY_FALLBACK = False
class FrenchAddressMatcher(LabelMatcher):
	def __init__(self):
		super(FrenchAddressMatcher, self).__init__(F_ADDRESS, COMMUNE_LEXICON, MATCH_MODE_EXACT)
	@timed
	def match(self, c):
		v = c.value_to_match()
		if len(v) < 1: return
		response = urllib.request.urlopen("http://api-adresse.data.gouv.fr/search/?q=" + urllib.parse.quote_plus(v))
		resp = response.read().decode('utf-8')
		data = json.loads(resp)
		if not data or 'features' not in data: return
		logging.debug('Returned %d results from api-adresse.data.gouv.fr for %s', len(data['features']), v)
		# Quick and dirty way to have two-tier results since based on BAN address matching results, when parsing a
		# coarse-grained entity (city or equivalent) the results at a finer level (street, etc.) are completely unreliable
		# as basically they are random, if not made-up, street addresses and districts.
		hits = [set(), set()]
		for point in data['features']:
			if 'properties' not in point: continue
			props = point['properties']
			if 'type' not in props:
				logging.warning('Properties do not contain any geolocation feature type! %s', data)
				continue
			kind = props['type']
			iIdx = dict() # Build inverted index to register partial matches
			if kind in ['housenumber', 'street', 'place', 'locality']: # Accurate enough, trust the result
				l = props['label']
				if l not in iIdx: iIdx[l] = props
				hits[1].add(l)
			elif kind in ['town', 'city', 'municipality']: # Sanity check on the commune name : exclude []
				v = normalize_and_validate_phrase(v)
				if v is not None and v in self.labelsMap:
					l = props['label']
					if l not in iIdx: iIdx[l] = props
					hits[0].add(l)
			else:
				logging.warning('Properties unexpected geolocation feature type: %s', kind)
		scoreFilter = partial(address_filter_score, v)
		prioHits = sorted(hits[0] if len(hits[0]) > 0 else hits[1], key = scoreFilter, reverse = True)
		for h in prioHits:
			if scoreFilter(h) <= 100 or h not in iIdx:
				continue
			props = iIdx[h]
			res_address = ' '.join([props[banField] for (ourField, banField) in BAN_MAPPING if banField in props])
			if len(res_address) > 3:
				self.register_full_match(c, self.t, None, res_address)
				return
		if COUNTRY_FALLBACK:
			self.register_full_match(c, F_COUNTRY, 100, 'France')
		else:
			raise TypeError('Could not parse any address field value for "{}"'.format(v))

def address_filter_score(src, ref):
	a1, a2 = case_phrase(src), case_phrase(ref)
	return fuzz.partial_ratio(a1, a2) + fuzz.ratio(a1, a2)

# Acronym handling

class AcronymMatcher(TypeMatcher):
	def __init__(self, minAcroSize = 4, maxAcroSize = 6):
		super(AcronymMatcher, self).__init__(F_ACRONYMS)
		self.minAcroSize = minAcroSize
		self.maxAcroSize = maxAcroSize
	@timed
	def match(self, c):
		for (acro, i) in self.acronyms_in_phrase(c.value):
			self.register_partial_match(c, '{} - {}'.format(F_ACRONYMS, c.f), 100, acro, (i, i + len(acro)))
	def acronyms_in_phrase(self, phrase):
		keepAcronyms = False
		tokens = normalize_and_validate_tokens(phrase, keepAcronyms)
		for acro in set(self.acronymize_tokens(tokens)):
			i = phrase.find(acro)
			if i < 0 or (i > 0 and phrase[i-1].isalpha()) or (i + len(acro) < len(phrase) and phrase[i + len(acro)].isalpha()):
				i = phrase.find('.'.join(acro))
			if i < 0: continue
			yield (acro, i)
	def acronymize_tokens(self, tokens):
		for i1 in range(0, len(tokens)):
			for i2 in range (i1 + self.minAcroSize, min(i1 + self.maxAcroSize, len(tokens))):
				tl = tokens[i1 : i2]
				yield ''.join([t[0] for t in tl]).upper()

class VariantExpander(TypeMatcher):
	def __init__(self, variantsMapFile, targetType, keepContext, domainType = None, scorer = tokenization_based_score):
		super(VariantExpander, self).__init__(targetType)
		self.domainType = domainType
		self.keepContext = keepContext # if true, then the main variant will be surrounded by original context in the normalized value
		self.variantsMap = file_to_variant_map(variantsMapFile) # map from original alternative variant to original main variant
		self.scorer = scorer
		self.tokenIdx = defaultdict(set) # map from alternative variant as joined-normalized-token-list to original alternative variant
		self.minTokens = 3
		self.maxTokens = DTC
		# map of alternative variant`s (including main or not!), from normalized string to list of original strings:
		phrasesMap = validated_lexical_map(self.variantsMap.keys(), tokenize = True)
		for (phrase, altVariants) in phrasesMap.items():
			tokens = phrase.split()
			l = len(tokens)
			if l < 1 or l > DTC: continue
			self.minTokens = min(self.minTokens, l)
			self.maxTokens = max(self.maxTokens, l)
			matchedVariantPhrase = ' '.join(tokens[:self.maxTokens])
			for altVariant in altVariants:
				self.tokenIdx[matchedVariantPhrase].add(altVariant)
				if altVariant not in self.variantsMap:
					raise RuntimeError('Alternative variant {} not found in variants map'.format(altVariant))
	@timed
	def match(self, c):
		if self.domainType is not None and self.domainType not in c.non_excluded_types():
			raise TypeError('{} only found excluded type in "{}"'.format(self, c))
		tokens = normalize_and_validate_tokens(c.value)
		if tokens is None:
			raise TypeError('{} found no non-trivial token sequence in "{}"'.format(self, c))
		found_one = False
		for k2 in range(self.maxTokens, 0, -1):
			for k1 in range(0, len(tokens) + 1 - k2):
				matchSrcTokens = tokens[k1:k1 + k2]
				matchRefPhrase = ' '.join(matchSrcTokens)
				if matchRefPhrase not in self.tokenIdx:
					continue
				for altVariant in self.tokenIdx[matchRefPhrase]:
					score = self.scorer(matchSrcTokens, tokens, matchRefPhrase, altVariant)
					v = case_phrase(c.value)
					i1 = v.find(tokens[k1])
					if i1 >= 0: i2 = v.find(tokens[k1 + k2 - 1], i1) if k2 > 1 else i1
					mainVariant = self.variantsMap[altVariant]
					logging.debug('%s matched on %s: "%s" expanded to main variant "%s"', self, matchRefPhrase, altVariant, mainVariant)
					normedValue = ''.join([v[:i1], mainVariant, v[i2:]]) if self.keepContext else mainVariant
					if i1 == 0 and k1 + k2 == len(tokens):
						self.register_full_match(c, self.t, score, mainVariant)
					elif i1 < 0 or i2 < 0:
						logging.warning('%s could not find tokens "%s ... %s" in original "%s"', self, tokens[k1], tokens[k1 + k2 - 1], v)
						self.register_full_match(c, self.t, score, mainVariant)
					else:
						span = (i1, i2 + len(tokens[k1 + k2 - 1]))
						self.register_partial_match(c, self.t, score, mainVariant, span)
					found_one = True
		if not found_one:
			raise ValueError('{} found no token sequence to expand in "{}"'.format(self, c))

# Misc utilities related to value normalization

def convert_codes(s):
	return reduce(lambda r, c: r.replace(c[0], c[1]), ["-–", "’'"], s)

def check_non_consecutive_subsequence(superStr, subStr):
	'''Checks if b is a non-consecutive subsequence of a, and returns the start and end indexes in a. '''
	a, b = convert_codes(superStr), convert_codes(subStr)
	pos = 0
	res = -1
	for i, ch in enumerate(a):
		if pos < len(b) and ch == b[pos]:
			if res < 0: res = i
			pos += 1
	return (res, i) if pos == len(b) else None

def non_zero_ratio_score(scores, minRatio = 10):
	r = sum([(100 if s > 0 else 0) for s in scores]) / len(scores)
	return r if r >= minRatio else 0

def header_matchers():
	''' Generates type matcher objects that can be applied to each column header in order to infer
		whether that column's type is the matcher's type (or alternatively a parent type or a child type).'''
	headerSims = defaultdict(set)
	for row in file_row_iter('header_names', '|'):
		r = list(map(lambda s: s.strip(), row))
		if len(r) < 1: continue
		yield HeaderMatcher(r[0], set(r))
		logging.info('Registered matcher for <%s> header with %d variants', r[0], len(r))

VALUE_MATCHERS = list()
@timed
def value_matchers():
	''' Lazy, one-time-only creation of value matchers list. '''
	if len(VALUE_MATCHERS) < 1:
		for vm in generate_value_matchers():
			VALUE_MATCHERS.append(vm)
	return VALUE_MATCHERS

def generate_value_matchers(lvl = 1):
	''' Generates type matcher objects that can be applied to each value cell in a column in order to infer
		whether that column's type is the matcher's type (or alternatively a parent type or a child type).

		Parameter:
		lvl 0 for lightweight matching, 2 for the heaviest variants, 1 as an intermediate level
		'''

	# Identifiers (typically but not necessarily unique)
	# yield TemplateMatcher('Identifiant', 90) # TODO distinguish unique vs. non-unique

	# Person names
	if lvl >= 0: 
		yield LabelMatcher(F_FIRST, PRENOM_LEXICON, MATCH_MODE_EXACT)
	if lvl >= 2: 
		# maxTokens set to 2 in order to deal with composite first names
		yield TokenizedMatcher(F_FIRST, PRENOM_LEXICON,
			maxTokens = 2, scorer = partial(tokenization_based_score, minSrcTokenRatio = 20, minSrcCharRatio = 10))
	if lvl >= 0: 
		yield LabelMatcher(F_LAST, PATRONYME_LEXICON, MATCH_MODE_EXACT)
	if lvl >= 2:
		yield TokenizedMatcher(F_TITLE, file_to_set('titre_appel') | file_to_set('titre_academique'), maxTokens = 1)
	if lvl >= 2:
		yield CustomPersonNameMatcher()
		fullNameValidators = { F_FIRST: lambda v: len(stripped(v)) > 1, F_LAST: lambda v: len(stripped(v)) > 2 }
		yield CompositeRegexMatcher(F_PERSON, PAT_FIRST_NAME, { F_FIRST: 1},
			ignoreCase = True, validators = fullNameValidators)
		yield CompositeRegexMatcher(F_PERSON, PAT_FIRST_LAST_NAME, { F_FIRST: 1, F_LAST: 2 },
			ignoreCase = True, validators = fullNameValidators)
		yield CompositeRegexMatcher(F_PERSON, PAT_LAST_FIRST_NAME, { F_FIRST: 2, F_LAST: 1 },
			ignoreCase = True, validators = fullNameValidators)
		yield CompositeRegexMatcher(F_PERSON, PAT_FIRSTINITIAL_LAST_NAME, { F_FIRST: 1, F_LAST: 2 },
			ignoreCase = False, validators = fullNameValidators)
		yield CompositeRegexMatcher(F_PERSON, PAT_LAST_FIRSTINITIAL_NAME, { F_FIRST: 2, F_LAST: 1 },
			ignoreCase = False, validators = fullNameValidators)
	yield CompositeMatcher(F_PERSON, [F_TITLE, F_FIRST])
	# Negate person name matches when it's a street name
	if lvl >= 0: 
		yield RegexMatcher(F_PERSON, "(rue|avenue|av|boulevard|bvd|bd|chemin|route|place|allee) .{0,10} (%s)" % PAT_FIRST_NAME,
			g = 1, ignoreCase = True, partial = True, neg = True)

	# Web stuff: Email, URL
	if lvl >= 0: 
		yield RegexMatcher(F_EMAIL, PAT_EMAIL)
	if lvl >= 0: 
		yield RegexMatcher(F_URL, PAT_URL)

	# Phone number
	if lvl >= 0: 
		yield CustomTelephoneMatcher()
	if lvl >= 2: 
		yield CustomTelephoneMatcher(partial = True)

	# Other individual IDs

	# from https://fr.wikipedia.org/wiki/Code_Insee#Identification_des_individus
	if lvl >= 0: 
		yield StdnumMatcher(F_NIR, stdnum.fr.nir.is_valid, stdnum.fr.nir.compact)
		yield StdnumMatcher(F_NIF, stdnum.fr.nif.is_valid, stdnum.fr.nif.compact)
		yield StdnumMatcher(F_TVA, stdnum.fr.tva.is_valid, stdnum.fr.tva.compact)
	# if lvl >= 0: 
	#     yield RegexMatcher(F_NIR, "[0-9]15")

	# Date-time
	if lvl >= 0: 
		yield RegexMatcher(F_YEAR, "19[0-9]{2}")
		yield RegexMatcher(F_YEAR, "20[0-9]{2}")
	if lvl >= 1:
		yield CustomDateMatcher()
	yield SubtypeMatcher(F_DATE, [F_YEAR])
	yield SubtypeMatcher(F_STRUCTURED_TYPE, [F_DATE, F_URL, F_EMAIL, F_PHONE])

	# MESR Domain

	## Recherche
	PAT_SIREN = "[0-9]{9}"
	PAT_SIRET = "[0-9]{14}"
	if lvl >= 0: 
		yield StdnumMatcher(F_SIREN, stdnum.fr.siren.is_valid, stdnum.fr.siren.compact)
		yield StdnumMatcher(F_SIRET, stdnum.fr.siret.is_valid, stdnum.fr.siret.compact)
	PAT_NNS = "[0-9]{9}[a-zA-Z]"
	if lvl >= 0: 
		yield RegexMatcher(F_NNS, PAT_NNS)
	PAT_UAI = "[0-9]{7}[a-zA-Z]"
	if lvl >= 0: 
		yield RegexMatcher(F_UAI, PAT_UAI)
	if lvl >= 0: 
		yield RegexMatcher(F_UMR, "UMR-?[ A-Z]{0,8}([0-9]{3,4})", g = 1, partial = True)
	# Negate dates for all thoses regex matches (which happen to match using our custom matcher, especially for UAI patterns)
	if lvl >= 0:
		yield RegexMatcher(F_DATE, PAT_SIREN, ignoreCase = True, neg = True)
		yield RegexMatcher(F_DATE, PAT_SIRET, ignoreCase = True, neg = True)

	if lvl >= 2: 
		# yield LabelMatcher(F_RD_STRUCT, file_to_set('structure_recherche_short.col'), MATCH_MODE_EXACT)
		yield LabelMatcher(F_RD_STRUCT, file_to_set('org_rnsr.col'), MATCH_MODE_EXACT)
	if lvl >= 2: 
		yield TokenizedMatcher(F_RD_PARTNER, file_to_set('partenaire_recherche_ANR.col') | file_to_set('partenaire_recherche_FUI.col') |
			file_to_set('institution_H2020.col'), maxTokens = 6)
	if lvl >= 2: 
		yield TokenizedMatcher(F_CLINICALTRIAL_COLLAB, file_to_set('clinical_trial_sponsor_collab.col'),
		maxTokens = 4)
	yield SubtypeMatcher(F_RD, [F_RD_STRUCT, F_RD_PARTNER, F_CLINICALTRIAL_COLLAB])

	if lvl >= 2:
		try:
			import gridding
			yield GridMatcher()
		except ImportError as ie:
			logging.warning('Could not import gridding module required by GridMatcher')

	## Enseignement et Enseignement Supérieur
	academy_labels = file_to_set('academie') # file_to_set('academie') | file_to_set('region')
	yield VocabMatcher(F_MESR, file_to_set('org_enseignement.vocab'), ignoreCase = True, partial = False)
	if lvl >= 0:
		yield RegexMatcher(F_ACADEMIE, "acad.mie", ignoreCase = True)
		yield LabelMatcher(F_ACADEMIE, academy_labels, MATCH_MODE_EXACT, stopWords = ['académie'])    # SIES/APB
		yield TokenizedMatcher(F_ACADEMIE, academy_labels, distinctCount = 3)
	if lvl >= 0: 
		yield VocabMatcher(F_ETAB_NOTSUP, file_to_set('etab_1er_2nd_degres.vocab'), ignoreCase = True, partial = False)
		yield VocabMatcher(F_ETAB_ENSSUP, file_to_set('etab_enssup.vocab'), ignoreCase = True, partial = False)
	if lvl >= 1:
		lycees_synMap = {
			'Lycée général et technologique' : 'Lycée',
			'Lycée polyvalent' : 'Lycée',
			'LP': 'Lycée professionnel',
			'Ecole primaire privée': 'Ecole primaire'
		} 
		lycees_stopWords = ['privé', 'privée', 'public', 'publique']
		yield LabelMatcher(F_ETAB_NOTSUP, file_to_set('etab_1er_2nd_degres'), MATCH_MODE_EXACT, 
			stopWords = STOP_WORDS + lycees_stopWords, synMap = lycees_synMap)
	if lvl >= 0: 
		yield VocabMatcher(F_ETAB_ENSSUP, file_to_set('etab_enssup.vocab'), ignoreCase = True, partial = True, 
			matcher = VariantExpander('etab_enssup.syn', targetType = F_ETAB_ENSSUP, keepContext = False, domainType = F_MESR))
	if lvl >= 1:
		yield TokenizedMatcher(F_ETAB_ENSSUP, file_to_set('etab_enssup'), maxTokens = 6)
	yield SubtypeMatcher(F_ETAB, [F_ETAB_NOTSUP, F_ETAB_ENSSUP])
	if lvl >= 1: 
		yield LabelMatcher(F_APB_MENTION, file_to_set('mention_licence_sise'), MATCH_MODE_EXACT)
	if lvl >= 2: 
		yield TokenizedMatcher(F_APB_MENTION, file_to_set('mention_licence_apb2017.col'), maxTokens = 5)
	if lvl >= 2: 
		yield TokenizedMatcher(F_RD_DOMAIN, file_to_set('domaine_recherche.col'), maxTokens = 4)

	yield SubtypeMatcher(F_MESR, [F_RD, F_APB_MENTION, F_RD_DOMAIN, F_ETAB, F_ACADEMIE])

	# Geo Domain
	if lvl >= 1: 
		yield FrenchAddressMatcher()

	if lvl >= 2: 
		try:        
			from postal.parser import parse_address
			yield CustomAddressMatcher()
		except ImportError as ie:
			logging.warning('Could not import libpostal module required by CustomAddressMatcher')
	if lvl >= 0: 
		yield RegexMatcher(F_ZIP, "[0-9]{5}")

	if lvl >= 0:
		yield LabelMatcher(F_COUNTRY, file_to_set('country'), MATCH_MODE_EXACT)
		yield VariantExpander('country_fr_en.syn', targetType = F_COUNTRY, keepContext = True)

	if lvl >= 2: 
		yield TokenizedMatcher(F_CITY, COMMUNE_LEXICON, maxTokens = 3, stopWords = STOP_WORDS_CITY)
	elif lvl >= 0: 
		yield LabelMatcher(F_CITY, COMMUNE_LEXICON, MATCH_MODE_EXACT, stopWords = STOP_WORDS_CITY, synMap = { 'st': 'saint' })
		yield RegexMatcher(F_CITY, "(commune|ville) +de+ ([A-Za-z /\-]+)", g = 1, ignoreCase = True, partial = True)



	if lvl >= 2:
		yield TokenizedMatcher(F_DPT, file_to_set('departement'), distinctCount = 7)
		yield TokenizedMatcher(F_REGION, file_to_set('region'), distinctCount = 3)
	elif lvl >= 0:
		yield LabelMatcher(F_DPT, file_to_set('departement'), MATCH_MODE_EXACT)
		yield LabelMatcher(F_REGION, file_to_set('region'), MATCH_MODE_EXACT)
	if lvl >= 1: 
		yield TokenizedMatcher(F_STREET, file_to_set('voie.col'), maxTokens = 2)
	yield CompositeMatcher(F_ADDRESS, [F_STREET, F_ZIP, F_CITY, F_COUNTRY])
	yield SubtypeMatcher(F_GEO, [F_ADDRESS, F_ZIP, F_CITY, F_DPT, F_REGION, F_COUNTRY])

	# Publications
	# From http://stackoverflow.com/questions/27910/finding-a-doi-in-a-document-or-page
	PAT_DOI = '\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b'
	if lvl >= 0: 
		yield RegexMatcher(F_DOI, PAT_DOI)
	PAT_ISSN = '\d{4}-\d{3}[\dxX]$'
	if lvl >= 0: 
		yield RegexMatcher(F_ISSN, PAT_ISSN)
	 # TODO see if we should add IdRef
	yield SubtypeMatcher(F_PUBLI_ID, [F_DOI, F_ISSN])
	if lvl >= 0:
		pubTitleLexicon = file_to_set('titre_revue')
		yield LabelMatcher(F_JOURNAL, pubTitleLexicon, MATCH_MODE_EXACT)
	if lvl >= 2: 
		yield TokenizedMatcher(F_JOURNAL, pubTitleLexicon, maxTokens = 5, scorer = partial(tokenization_based_score, minSrcTokenRatio = 90))
	if lvl >= 2:
		articleLexicon = file_to_set('article_fr') | file_to_set('article_en')
		yield TokenizedMatcher(F_ARTICLE, articleLexicon)
	yield SubtypeMatcher(F_ARTICLE, [F_ABSTRACT, F_PUBLI_ID, F_ARTICLE_CONTENT])
	# yield BiblioMatcher()
	yield SubtypeMatcher(F_PUBLI, [F_JOURNAL, F_PUBLI_ID, F_ARTICLE, F_ABSTRACT])

	# Text fields
	yield SubtypeMatcher(F_TEXT, [F_FRENCH, F_ENGLISH])
	yield SubtypeMatcher(F_TEXT, [F_ARTICLE, F_ABSTRACT])

	# Biomedical Domain
	if lvl >= 2:
		yield TokenizedMatcher(F_CLINICALTRIAL_NAME, file_to_set('clinical_trial_acronym'))
		yield TokenizedMatcher(F_MEDICAL_SPEC,
			file_to_set('specialite_medicale_fr') |
			file_to_set('specialite_medicale_en'))
	yield SubtypeMatcher(F_BIOMEDICAL, [F_CLINICALTRIAL_NAME, F_MEDICAL_SPEC])

	# Agro Domain
	if lvl >= 0:
		phytoLexicon = file_to_set('phyto')
		yield LabelMatcher(F_PHYTO, phytoLexicon, MATCH_MODE_EXACT)
	if lvl >= 2: yield TokenizedMatcher(F_PHYTO, phytoLexicon,
		maxTokens = 4, scorer = partial(tokenization_based_score, minSrcTokenRatio = 30))
	yield SubtypeMatcher(F_AGRO, [F_PHYTO])

	# A few subsumption relations
	yield SubtypeMatcher(F_ORG_ID, [F_SIREN, F_SIRET, F_NNS, F_UAI, F_UMR])
	yield SubtypeMatcher(F_PERSON_ID, [F_PERSON, F_EMAIL, F_PHONE, F_NIR])
	yield SubtypeMatcher(F_ID, [F_ORG_ID, F_PERSON_ID, F_PUBLI_ID])

	# The top-level data type for organizations
	yield SubtypeMatcher(F_INSTITUTION, [F_RD_STRUCT, F_ETAB, F_ENTREPRISE])

	# Normalize by expanding alternative variants (such as acronyms, abbreviations and synonyms) to their main variant
	if lvl >= 0:
		yield VocabMatcher(F_RD_STRUCT, file_to_set('org_RD.vocab'), ignoreCase = True, partial = False,
			matcher = VariantExpander('org_hal.syn', targetType = F_RD_STRUCT, keepContext = True))
		yield VocabMatcher(F_ENTREPRISE, file_to_set('org_entreprise.vocab'), ignoreCase = True, partial = False,
			matcher = VariantExpander('org_entreprise.syn', targetType = F_ENTREPRISE, keepContext = True))
		yield VocabMatcher(F_EDUC_NAT, file_to_set('educ_nat.vocab'), ignoreCase = True, partial = False,
			matcher = VariantExpander('educ_nat.syn', targetType = F_EDUC_NAT, keepContext = False))

	if lvl >= 0:
		yield VocabMatcher(F_RD_STRUCT, file_to_set('org_rnsr.vocab'), ignoreCase = True, partial = False,
			matcher = VariantExpander('org_rnsr.syn', targetType = F_RD_STRUCT, keepContext = True, domainType = F_RD_DOMAIN))

	# Spot acronyms on-the-fly
	if lvl >= 1: 
		yield AcronymMatcher()

def all_data_types():
	''' Returns a map of categories (including a default one for orphan data types) to a list of data types 
	'''
	allTypes = set([vm.t for vm in value_matchers()])
	categorizedTypes = defaultdict(list)
	for parent, children in PARENT_CHILD_RELS.items():
		for child in children:
			# We allow for duplicates under different categories
			if child not in categorizedTypes[parent]:
				categorizedTypes[parent].append(child)
			allTypes.discard(child)
	categorizedTypes[C_OTHERS] = list(allTypes)
	return categorizedTypes

def type_tags():
	allTags = defaultdict(list)
	allTags.update(TYPE_TAGS)
	for mainTag in allTags.keys():
		allTags[mainTag].append(mainTag)
	for parent, children in PARENT_CHILD_RELS.items():
		for parentTag in allTags[parent]:
			for child in children:
				if parentTag not in allTags[child]:
					allTags[child].append(parentTag)
	return allTags

# Main functionality

def pre_process_headers(inferences, m):
	scores = list()
	for (f, i) in inferences.items():
		s = m.scoreHeaderValue(f, i)
		if s > 0: scores.append((i, s, f))
	if len(scores) < 1:
		logging.info('NO HEADER MATCH on <%s>', m)
	elif len(scores) > 1:
		logging.info('MULTIPLE HEADER MATCHES on <%s>, ignoring them', m)
	else:
		scores[0][0].register(scores[0][1], m.t, str(m), f = scores[0][2])

def post_process_headers(inferences, m):
	scores = list()
	for (f, i) in inferences.items():
		s = m.scoreHeaderValue(f, i)
		if s > 0: scores.append((i, s))
	if len(scores) < 1:
		logging.info('NO HEADER MATCH on <%s>', m)
	elif len(scores) > 1:
		logging.info('MULTIPLE HEADER MATCHES on <%s>, ignoring them', m)
	else:
		logging.info('SINGLE HEADER MATCH on <%s>', m)
		scores[0][0].boost(scores[0][1], m.t, .5)

### API method implementations (using Pandas DataFrames as input/output)

def infer_types(tab, params = None):
	'''  Infers column types for the input array and produces a dictionary of column name to likeliest types. '''
	fields = parse_fields_from_Panda(tab)
	return { 
		'column_types': fields.infer_types(), 
		'all_types': all_data_types(),
		'type_tags': type_tags() }

def normalize_values(tab, params):
	''' Normalizes the values in each column whose type has been identified, and returns the input array after adding the
		resulting new columns.

		Based on those values, more than one additional column may be added to a given input field:
		- extracted components for a composite type
		- variants for a data type within a domain rich in lexical variations like synonyms, etc. '''
	modified = pd.DataFrame(False, index=tab.index, columns=tab.columns)
	fields = parse_fields_from_Panda(tab)
	# Fetch results of previous step (type inference) so as to avoid duplicative work
	trusted_types = params['column_types']
	fields.match_by_types(trusted_types)
	for (originalField, newCol) in fields.normalize_values_in_place(trusted_types):
		modified[originalField] = (tab[originalField] != newCol)
		tab.loc[:, originalField] = newCol
	return tab, modified

def sample_types_ilocs(tab, params, sample_params):
	num_rows_to_display = sample_params.get('num_rows_to_display', 30)
	randomize = sample_params.get('randomize', True)
	idxs = np.random.permutation(range(num_rows_to_display)) if randomize else range(num_rows_to_display)
	row_idxs = idxs[:num_rows_to_display]
	return row_idxs

### Main method

if __name__ == '__main__':
	logging.basicConfig(filename = 'log/preprocess_fields.log', level = logging.DEBUG)
	print('Python version: {}'.format(sys.version))
	parser = optparse.OptionParser()
	parser.add_option("-s", "--src", dest = "srcFileName",
						help = "source file")
	parser.add_option("-d", "--delimiter", dest = "delimiter",
						help = "CSV delimiter")
	parser.add_option("-f", "--output_format", dest = "of",
						help = "Output format (md / csv)")
	parser.add_option("-p", "--in_place", dest = "ip",
						help = "in-place normalization")
	(options, args) = parser.parse_args()
	separator = options.delimiter if options.delimiter else '|'
	outputFormat = options.of if options.of else separator
	fields = parse_fields_from_CSV(options.srcFileName, delimiter = separator)
	inPlace = False # options.ip

	# Single-pass method
	# fields.process_values(outputFormat = outputFormat)

	if inPlace:
		# In-place normalization
		types = fields.infer_types()
		for (field, row) in fields.normalize_values_in_place(types):
			for value in row: print(value)
		sys.exit()

	# With addition of new fields
	res = list()
	for i in range(fields.entries): res.append(dict())
	types = fields.infer_types()
	hofs = list()
	for (of, vs) in fields.normalize_values(types):
		hofs.append(of)
		for i, v in enumerate(vs):
			res[i][of] = v
	# Output normalization results
	ofs = sorted(uniq(hofs))
	if outputFormat == 'md':
		print('|{}|'.format('|'.join(ofs)))
		print('|{}|'.format('|'.join('-' * len(ofs))))
	else:
		print(outputFormat.join(ofs))
	for row in res:
		if outputFormat == 'txt':
			for k, v in row.items():
				if not k.startswith('++'): print(k.rjust(60), str(v))
			for k, v in row.items():
				if k.startswith('++'): print(k.rjust(60), str(v))
			print()
		elif outputFormat == 'md':
				print('|{}|'.format('|'.join([str(row[f]) if f in row else '' for f in ofs])))
		else:
				print(outputFormat.join([str(row[f]) if f in row else '' for f in ofs]))
	for key in timingInfo:
		logging.info('TIMING for {}: {} calls, cumulated {} ms'.format(key, countInfo[key], int(timingInfo[key] / 1000)))
	# Output run info
	columnsReplaced = dict()
	allFields = []
	for (h, f) in fields.fields.items():
		fieldName = h.value
		allFields.append(fieldName)
		columnsReplaced[fieldName] = sum([fields.modifiedByColumn[fieldName][i] > 0 for i in range(fields.entries)])
	print('Replaced by columns', columnsReplaced)
	allReplaced = sum([any([i < len(fields.modifiedByColumn[fieldName]) and fields.modifiedByColumn[fieldName][i] for fieldName in allFields]) for i in range(fields.entries)])
	print('Total replaced', allReplaced)
