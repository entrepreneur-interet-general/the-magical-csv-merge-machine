#!/usr/bin/env python3
# coding=utf-8

import re, unicodedata
from collections import defaultdict
from functools import reduce

# Natural language toolbox and lexical stuff: FRENCH ONLY!!!

STOP_WORDS = [
	# Prepositions (excepted "avec" and "sans" which are semantically meaningful)
	"a", "au", "aux", "de", "des", "du", "par", "pour", "sur", "chez", "dans", "sous", "vers", 
	# Articles
	"le", "la", "les", "l", "c", "ce", "ca", 
	 # Conjonctions of coordination
	"mais", "et", "ou", "donc", "or", "ni", "car"
]

def isStopWord(word): return word in STOP_WORDS

# Utilities

def replaceBySpace(str, *patterns): return reduce(lambda s, p: re.sub(p, ' ', s), patterns, str)

def toASCII(phrase): return unicodedata.normalize('NFKD', phrase)

def preSplit(v):
	s = ' ' + v.strip() + ' '
	s = replaceBySpace(s, '[\{\}\[\](),\.\"\';:!?&\^\/\*-]')
	return re.sub('([^\d\'])-([^\d])', '\1 \2', s)

def splitAndCase(phrase):
	return list([caseToken(t) for t in str.split(preSplit(phrase))])

def validateTokens(phrase, isFirst):
	if phrase:
		tokens = splitAndCase(phrase)
		validTokens = []
		for token in tokens:
			valid = isValidFirstnameToken(token) if isFirst else isValidSurnameToken(token)
			if valid: validTokens.append(token)
		if len(validTokens) > 0 and len(validTokens) == len(tokens): return validTokens
	return []

def isValidSurnameToken(token):
	token = stripped(token)
	if token.isspace() or not token: return False
	if token.isdigit(): return False
	if len(token) <= 2 and not (token.isalpha() and token.isupper()): return False
	return not isStopWord(token)

def isValidFirstnameToken(token):
	token = stripped(token)
	if token.isspace() or not token: return False
	return not token.isdigit()

def normalizeAndValidatePhrase(phrase, isFirst):
	tokens = validateTokens(phrase, isFirst)
	if len(tokens) > 0:
		isValid = (len(tokens) == 1 or all([len(t) == 1 for t in tokens])) if isFirst else ((len(tokens) == 1 and len(tokens[0]) == 1) or all([len(t) > 1 for t in tokens]))
		if isValid: return ' '.join(tokens)
	return None

def caseToken(t): return toASCII(t.strip().lower())

def validatedLexiconMap(lexicon, isFirst, tokenize = False): 
	''' Returns a dictionary from normalized string to list of original strings. '''
	lm = defaultdict(list)
	for s in lexicon:
		k = normalizeAndValidatePhrase(s, isFirst) if tokenize else caseToken(s)
		if k is None: continue
		lm[k].append(s)
	return lm

#####

def stripped(s): return s.strip(" -_.,'?!").strip('"').strip()
def fileToList(fileName): 
	with open(fileName, mode = 'r') as f: 
		return [stripped(line) for line in f]
def fileToSet(fileName): return set(fileToList(fileName))

# Custom parsing of person names

F_FIRST = u'Prénom'
F_LAST = u'Nom'
F_TITLE = u'Titre'
F_FIRSTORLAST = F_FIRST + '|' + F_LAST

PRENOM_LEXICON = fileToSet('resource/prenom')
PATRONYME_LEXICON = fileToSet('resource/patronyme_fr')

FR_FIRSTNAMES = set([s.lower() for s in PRENOM_LEXICON])
FR_SURNAMES = set([s.lower() for s in PATRONYME_LEXICON])

PN_STRIP_CHARS = ' <>[](){}"\''
PN_INSIDE_DELIMITER = ','
PN_OUTSIDE_DELIMITERS = ';+/'
FN_INITIAL_SEPS = '-.'
FN_COMP_DELIMITER = '-'

TITLE_MONSIEUR = 'M.'
PN_TITLE_VARIANTS = {
	TITLE_MONSIEUR: [
	'mr', 'monsieur', 'mister', 'sir'], # Form "M." is handled separately
	'Mme': ['mme', 'mrs', 'madame', 'madam', 'ms', 'miss'],
	'Dr': ['dr', 'docteur', 'doctor'],
	'Pr': ['pr', 'professeur', 'prof', 'professor']
}

def splitAndKeepDelimiter(s, delimiter):
    return reduce(lambda l, e: l[:-1] + [l[-1] + e] if e == delimiter else l + [e], re.split("(%s)" % re.escape(delimiter), s), [])

def nameTokenizer(s, isFirst):
	for token in s.split():
		for t1 in splitAndKeepDelimiter(token, FN_COMP_DELIMITER) if isFirst else [token]:
			for t2 in splitAndKeepDelimiter(t1, '.'):
				if t2 not in FN_INITIAL_SEPS: yield t2

def joinChoices(s): 
	l = list(s)
	if len(l) < 1: return None
	return l[0] + ('({})'.format(' or '.join(l[1:])) if len(l) > 1 else '')

def joinPersonName(d):
	l = [joinChoices([ln.upper() for ln in d[F_LAST]])]
	if F_FIRST in d and len(d[F_FIRST]) > 0: l = [joinChoices([capitalizeCompound(fn) + ('.' if len(fn) == 1 else '') for fn in d[F_FIRST]])] + l
	if F_TITLE in d and len(d[F_TITLE]) > 0: l = [d[F_TITLE]] + l
	return ' '.join(l)

def capitalizeCompound(fn):
	return FN_COMP_DELIMITER.join([c.capitalize() for c in fn.split(FN_COMP_DELIMITER)])

def extractPersonNames(l):
	for d in PN_OUTSIDE_DELIMITERS:
		cs = l.split(d)
		if len(cs) > 1 and any([PN_INSIDE_DELIMITER in c for c in cs]):
			return extractPersonNames(PN_INSIDE_DELIMITER.join([' '.join(reversed(c.split(PN_INSIDE_DELIMITER))) for c in cs]))
	s = re.sub(r'[\b\s](et|and|&)[\b\s]', PN_INSIDE_DELIMITER, l, flags=re.IGNORECASE)
	s0 = s.translate({ PN_STRIP_CHARS: None })
	(s1, s2, s3) = s0.partition(PN_INSIDE_DELIMITER)
	if len(s2) == 1: 
		if PN_INSIDE_DELIMITER not in s1 + s3:
			b1, a3 = extractLastName(s1), extractAnyFirstName(s3)
			if b1 and a3: 
				return [{ F_FIRST: set([a3]), F_LAST: [b1] }]
			a1, b3 = extractAnyFirstName(s1), extractLastName(s3)
			if a1 and b3: 
				return [{ F_FIRST: set([a1]), F_LAST: [b3] }]
		l1, l3 = extractPersonNames(s1), extractPersonNames(s3)
		if len(l1) + len(l3) > 0: return l1 + l3
	return personNameSingleton(s0)

def personNameSingleton(s): 
	d = extractPersonName(s)
	return [] if d is None else [d]

def isInitial(name): 
	return not all([(len(s) == 1 or (len(s) == 2 and s[1] in FN_INITIAL_SEPS)) for s in nameTokenizer(name, True)])

def appendFirst(firstComps, alternating, fis, d, i, t):
	if (isInitial(t) and not all([isInitial(firstComp[1]) for firstComp in firstComps])) or (not isInitial(t) and all([isInitial(firstComp[1]) for firstComp in firstComps])):
		recordFirst(firstComps, alternating, fis, d)
	firstComps.append((i, t))

def recordFirst(firstComps, alternating, fis, d):
	if len(firstComps) < 1: return
	elif len(firstComps) == 1: 
		fp = firstComps[0]
		fst = fp[1]
		d[F_FIRST].add(fst)
		fis.add((fp[0], fp[0]))
		d[F_FIRSTORLAST].append(fst)
		alternating.append((1, fst))
	else:
		fst = FN_COMP_DELIMITER.join([firstComp[1] for firstComp in firstComps])
		d[F_FIRST].add(fst)
		fis.add((firstComps[0][0], firstComps[-1][0]))
		alternating.append((0, fst))
	firstComps.clear()

LABELS_MAP = validatedLexiconMap(PRENOM_LEXICON, True)

def extractFirstName(s):
	v = normalizeAndValidatePhrase(s, True)
	return v if (v is not None and v in LABELS_MAP) else None

def extractAnyFirstName(s):
	tokens = list(nameTokenizer(s, True))
	comps = list()
	for s in tokens:
		if s[-1] in FN_INITIAL_SEPS:
			t = s.strip(FN_INITIAL_SEPS)
			if t.isalpha(): comps.append(t)
		else:
			t0 = s.strip(FN_INITIAL_SEPS)
			if len(t0) > 1:
				t = t0.lower()
				comps.append(t)
	if len(comps) > 0:
		skipped = len(comps) < len(tokens)
		isValid = len(comps) == 1 or (all([len(t) == 1 for t in comps]) and not skipped)
		if isValid: return FN_COMP_DELIMITER.join(comps)
	return None

def extractLastName(s):
	return normalizeAndValidatePhrase(s, False)

def extractPersonName(s):
	d = { F_FIRST: set(), F_FIRSTORLAST: list(), F_LAST: list() }
	# Set of first-name (start token, end token) pairs representing the spans 
	fis = set() 
	tokens = list(nameTokenizer(s, True))
	firstComps = list()
	# List of pairs (kind, component) where kind is 0 for a first-name component, 1 for first or surname, 2 for surname
	alternating = list() 
	firstDone, lastDone = False, False
	for i, token in enumerate(tokens):
		t0 = token.strip(FN_INITIAL_SEPS)
		if token[-1] in FN_INITIAL_SEPS:
			if not t0.isalpha(): continue
			t = token if firstDone else token.strip(FN_INITIAL_SEPS)
			if i == 0 and t in PN_TITLE_VARIANTS[TITLE_MONSIEUR]: d[F_TITLE] = TITLE_MONSIEUR
			if not firstDone: appendFirst(firstComps, alternating, fis, d, i, t.lower())
			elif not lastDone: d[F_LAST].append(t.lower())
			continue
		if len(t0) < 2: continue
		t = t0.lower()
		title = None
		for (main, variants) in PN_TITLE_VARIANTS.items():
			if t0[1:].islower() and t in [v.lower() for v in variants]:
				title = main
				break
		if title: 
			recordFirst(firstComps, alternating, fis, d)
			d[F_TITLE] = title
			continue
		if t in FR_FIRSTNAMES:
			appendFirst(firstComps, alternating, fis, d, i, t)
			firstDone = True
		else:
			t1 = t.strip(FN_INITIAL_SEPS)
			recordFirst(firstComps, alternating, fis, d)
			d[F_LAST].append(t)
			lastDone = True
			alternating.append((2, t))
	recordFirst(firstComps, alternating, fis, d)
	if len(d[F_LAST]) < 1:
		if len(d[F_FIRSTORLAST]) > 0:
			d[F_LAST] = list(d[F_FIRSTORLAST])
			for f in d[F_FIRST]: 
				if f in  d[F_LAST]: d[F_LAST].remove(f)
			d[F_FIRST] = d[F_FIRST] - set(d[F_LAST])
			d[F_FIRSTORLAST] = list(d[F_LAST]) 
			for f in d[F_FIRST]:
				if f in  d[F_FIRSTORLAST]: d[F_FIRSTORLAST].remove(f)
	if len(d[F_FIRST]) > 1 or len(d[F_LAST]) > 1:
		# FIXME unstack sequence if it's first/last-alternating...
		pass
	if len(d[F_LAST]) < 1:
		# FIXME here something like "C. Léonard" takes Léonard as first and not as last, this is wrong
		for candidates in candidatesList(tokens, fis, d):
			if len(candidates) > 0: 
				d[F_LAST].append(sorted(candidates, key = lambda c: len(c), reverse = True)[0])
				break
	if len(d[F_LAST]) < 1:
		pickBestFirstAsLast(d)
	if len(d[F_LAST]) < 1: return None
	if len(d[F_FIRST]) < 1:
		d[F_FIRST] = set(d[F_FIRSTORLAST]) - set(d[F_LAST])
	if F_FIRSTORLAST in d: del d[F_FIRSTORLAST]
	if TITLE_MONSIEUR[:-1] in d[F_FIRST]: del d[F_TITLE]
	noLastAsFirstComponent(d)
	new_last = list()
	components = []
	for ln in d[F_LAST]:
		if ln.endswith(FN_COMP_DELIMITER):
			components.append(ln)
		elif len(components) < 1:
			new_last.append(norm(ln))
		else:
			components.append(ln)
			new_last.append(FN_COMP_DELIMITER.join([norm(c) for c in components]))
			components.clear()
	if len(components) > 0:
		new_last.append(''.join([norm(c) for c in components]))
	d[F_LAST] = list([nl if len(nl) > 1 else nl + '.' for nl in new_last])
	return d

def pickBestFirstAsLast(d):
	if len(d[F_FIRST]) == 1:
		fn = list(d[F_FIRST])[0]
		if not isInitial(fn):
			d[F_LAST].append(fn)
			d[F_FIRST].remove(fn)
	elif len(d[F_FIRST]) > 1:
		bestLNs = sorted([fn for fn in d[F_FIRST] if fn.lower() not in FR_FIRSTNAMES], key = lambda s: len(s), reverse = True)
		if len(bestLNs) > 0:
			d[F_LAST].append(bestLNs[0])
			d[F_FIRST].remove(bestLNs[0])

def noLastAsFirstComponent(d):
	toRemove = set()
	toAdd = set()
	for fn in d[F_FIRST]:
		firstComps = norm(fn).split(FN_COMP_DELIMITER)
		firstComps0 = set(firstComps) & set([norm(ln) for ln in d[F_LAST]])
		if len(firstComps0) > 0:
			toRemove.add(fn)
			toAdd.add(FN_COMP_DELIMITER.join([c for c in firstComps if norm(c) not in firstComps0]))
	d[F_FIRST] -= toRemove
	d[F_FIRST] |= toAdd

def norm(c): return c.strip('-.,').upper()

def noFirstAsLast(token, d): return not any([norm(token).find(norm(fn)) >= 0 for fn in d[F_FIRST]])

def candidatesList(allTokens, fis, d):
	tokens = [t.strip(FN_INITIAL_SEPS) for t in allTokens]
	# Start with neighbors of the first name token matches
	yield set(filter(lambda token: noFirstAsLast(token, d), [tokens[fi[0] - 1] for fi in fis if fi[0] > 0]))
	yield set(filter(lambda token: noFirstAsLast(token, d), [tokens[fi[1] + 1] for fi in fis if fi[1] < len(tokens) - 1]))
	# Then prioritize upper-case tokens not included in any first name
	yield set(filter(lambda token: len(token) > 2 and token.isupper() and noFirstAsLast(token, d), tokens))
	# Then capitalized tokens not included in any first name
	yield set(filter(lambda token: len(token) > 2 and token.capitalize() == token and noFirstAsLast(token, d), tokens))
	# Then relax first-name constraint to full string comparison
	yield set(filter(lambda token: len(token) > 2 and token.isupper() and not any([token == fn.upper() for fn in d[F_FIRST]]), tokens))
	yield set(filter(lambda token: len(token) > 2 and token.capitalize() == token and not any([token == fn.capitalize() for fn in d[F_FIRST]]), tokens))
	# Then neglect casing...
	yield set(filter(lambda token: len(token) > 2 and not any([token.upper() == fn.upper() for fn in d[F_FIRST]]), tokens))

# Top-level parsing methods

def customParsePersonNamesAsStrings(l): return map(joinPersonName, extractPersonNames(l))

MD_OUTPUT_MAX = 3

def printCustomParsePersonNamesAsHeader(): 
	print('## Résultats de normalisation de noms de personnes')
	print('')
	print('|{}|'.format('|'.join(['Source'] + ['Nom #' + str(i + 1) for i in range(MD_OUTPUT_MAX)])))
	print('|{}|'.format('|'.join('-' * (MD_OUTPUT_MAX + 1))))

def printCustomParsePersonNamesAsMd(l): 
	ls = list(customParsePersonNamesAsStrings(l))
	print('|{}|'.format('|'.join([l] + ls[:3])))

if __name__ == '__main__':
	printCustomParsePersonNamesAsHeader()
	for (src, refs) in [
		('J.P. Poly', [('J.P.', 'Poly')]),
		('Justine, J.-L.', [('Justine', 'J.-L.')]),
		('Thibault, André', [('André', 'Thibault')]),
		('Véronique Wester-Ouisse', [('Véronique', 'Wester-Ouisse')]),
		('Abdellah Bounfour, Kamal Naït-Zerrad et Abdallah Boumalk', [('Abdellah', 'Bounfour'), ('Kamal', 'Naït-Zerrad'), ('Abdallah', 'Boumalk')]),
		('Schreck E., Gontier L. and Treilhou M.', [('E.', 'Schreck'), ('L.', 'Gontier'), ('M.', 'Treilhou')]),
		('L. Jutier, C. Léonard and F. Gatti.', [('L.', 'Jutier'), ('C.', 'Léonard'), ('F.', 'Gatti.')]),
		('Adolphe L.', [('L.', 'Adolphe')]),
		('Pierre-André MIMOUN', [('Pierre-André', 'MIMOUN')]),
		('Pierre-André mimoun', [('Pierre-André', 'MIMOUN')]),
		('Alain Schnapp', [('Alain', 'Schnapp')]),
		('Schnapp Alain', [('Alain', 'Schnapp')]),
		('BADIE Bertrand', [('Bertrand', 'BADIE')]),
		('BAKHOUCHE Béatrice', [('Béatrice', 'BAKHOUCHE')]),
		('Emmanuel WALLON (sous la direction de)', [('Emmanuel', 'WALLON')]),
		('Charles-Edmond BICHOT', [('Charles-Edmond', 'BICHOT')]),
		('Sylvie Neyertz et David Brown', [('Sylvie', 'Neyertz'), ('David', 'Brown')]),
		('Anne-Dominique Merville & Antoine COPPOLANI', [('Anne-Dominique', 'Merville'), ('Antoine', 'COPPOLANI')]),
		('Dominique Kalifa', [('Dominique', 'Kalifa')]),
		('S. CHAKER (Dir.)', [('S.', 'CHAKER')])
	 ]:
	 	printCustomParsePersonNamesAsMd(src)
	# with open('test_data/person_names.to_normalize', mode = 'r') as f: 
	# 	for line in f:
	# 		printCustomParsePersonNamesAsMd(stripped(line))			
