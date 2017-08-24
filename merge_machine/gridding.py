import csv, difflib, functools, logging, re, os, sys
import unicodedata, unidecode
from collections import defaultdict, Counter

from fuzzywuzzy import fuzz
from postal.parser import parse_address
import enchant

DICTS = [ enchant.Dict("en_US") ]
SOURCE = 1
REFERENCE = 2
MIN_STRING_SCORE = 20
REQUIRES_SHARED_PROPER_NOUN = True
DONT_DISCRIMINATE = False
EXCLUDE_ADDRESS_AS_LABEL = False
FORBID_DUPE_ACRONYMS = False

def acronymizeTokens(tokens, minAcroSize = 3, maxAcroSize = 7):
	for s in range(max(minAcroSize, len(tokens)), min(maxAcroSize, len(tokens)) + 1):
		tl = tokens[:s]
		yield (''.join([t[0] for t in tl]).upper(), tl)

def acronymizePhrase(phrase, keepAcronyms = True): 
	tokens = validateTokens(stripped(phrase), keepAcronyms)
	return list(acronymizeTokens(tokens))

ACRO_PATTERNS = ['[{}]', '({})']
def findValidAcronyms(phrase):
	for acro in acronymizePhrase(phrase, True):
		if any([phrase.find(ap.format(acro)) >=0 for ap in ACRO_PATTERNS]): 
			yield acro

UR_REGEX_GRP = '([A-Z0-9]{2,3}[0-9])' # '([A-Z]?[0-9]{3,4})'
UR_REGEXES_LABEL = dict([(kind, '\\b' + kind + ' ?-? ?' + UR_REGEX_GRP + '\\b') for kind in ['UR', 'UFR', 'UMR', 'UPR', 'CNR', 'EA', 'CNRS']])
UR_REGEXES_URL = dict([(kind, '\\b' + kind + UR_REGEX_GRP + '\\b') for kind in ['UR', 'UFR', 'UMR', 'UPR', 'CNR', 'EA', 'CNRS']])
REQUIRED_ADDR_FEATURES = set(['road', 'city', 'country'])

def regex_variant(kind, m): 
	umr = m.replace(' ', '').replace('-', '')
	return '{} {}'.format(kind, int(umr) if umr.isdigit() else umr)

def enrich_item_with_variants(label):
	res = dict(label = label, categories = set(), variants = set())
	# Acronyms
	for (acro, variant) in extractAcronymsByColocation(label):
		item['variants'].add(variant)
		item['acros'].add(acro)
	# Addresses (French or foreign)
	addr = parse_address(label)
	features = dict((f, v) for (v, f) in addr)
	if len(REQUIRED_ADDR_FEATURES | features.keys()) > 0:
		res['address_as_label'] = label
		if 'city' in features: res['city'] = features['city']
		if 'country' in features: res['country'] = features['country']
	# Unité Mixte de Recherche and such things
	for (kind, regex) in UR_REGEXES_LABEL.items():
		ms = re.findall(regex, label)
		if ms:
			for m in ms:
				variant = regex_variant(kind, m)
				logging.info('Found UMR-type match: {} in label "{}"'.format(variant, label))
				res['variants'].add(variant)
				res['ur_id'] = variant
	# Categorization
	tokens = validateTokens(label)
	for token in tokens:
		cat = categorize(token)
		if cat is not None: res['categories'].add(cat)
	# Duplicated tokens
	i = label.find('')
	if i > 0:
		pre = justCase(label[:i])
		post = justCase(label[i+1])
		if post.startswitch(pre): res['variants'].add(post)

def extractAcronymsByConstruction(phrase):
	for acro in acronymizePhrase(phrase, True):
		for ap in ACRO_PATTERNS:
			substring = ap.format(acro)
			i = phrase.find(substring)
			if i >=0:
				yield (acro, phrase[:i] + phrase[i + len(substring):])

def extractAcronymsByColocation(phrase):
	ms = re.findall('\b\[([\s0-9A-Z/]+)\]\b', phrase)
	if not ms: return
	for m in ms:
		yield (m.group(1), phrase[:m.start()])

def isStopWord(word): return word in STOP_WORDS

def isValidPhrase(tokens): return len(tokens) > 0 and not all(len(t) < 2 and t.isdigit() for t in tokens)

def stripped(s): return s.strip(" -_.,'?!*+").strip('"').strip()

def isValidToken(token, minLength = 2):
	token = stripped(token)
	if token.isspace() or not token: return False
	if token.isdigit(): return False # Be careful this does not get called when doing regex or template matching!
	if len(token) <= minLength and not (token.isalpha() and token.isupper()): return False
	return not isStopWord(token)

def isValidValue(v):
	''' Validates a single value (sufficient non-empty data and such things) '''
	stripped = stripped(v)
	return len(stripped) > 0 and stripped not in ['null', 'NA', 'N/A']

def isAcroToken(token):
	return re.match("[A-Z][0-9]*$", token) or re.match("[A-Z0-9]+$", token)

MIN_ACRO_SIZE = 3
MAX_ACRO_SIZE = 6

def lowerOrNot(token, keepAcronyms, keepInitialized = False):
	''' Set keepAcronyms to true in order to improve precision (e.g. a CAT scan will not be matched by a kitty). '''
	if keepAcronyms and len(token) >= MIN_ACRO_SIZE and len(token) <= MAX_ACRO_SIZE and isAcroToken(token):
		return token
	if keepInitialized:
		m = re.search("([A-Z][0-9]+)[^'a-zA-Z].*", token)
		if m:
			toKeep = m.group(0)
			return toKeep + lowerOrNot(token[len(toKeep):], keepAcronyms, keepInitialized)
	return token.lower()

def toASCII(phrase): return unidecode.unidecode(phrase)

def caseToken(t, keepAcronyms = False): return toASCII(lowerOrNot(t.strip(), keepAcronyms))

def replaceBySpace(str, *patterns): return functools.reduce(lambda s, p: re.sub(p, ' ', s), patterns, str)

def dehyphenateToken(token):
	result = token[:]
	i = result.find('-')
	while i >= 0 and i < len(result) - 1:
		left = result[0:i]
		right = result[i+1:]
		if left.isdigit() or right.isdigit(): break
		result = left + right
		i = result.find('-')
	return result.strip()

def preSplit(v):
	s = ' ' + v.strip() + ' '
	s = replaceBySpace(s, '[\{\}\[\](),\.\"\';:!?&\^\/\*-]')
	return re.sub('([^\d\'])-([^\d])', '\1 \2', s)

def justCase(phrase, keepAcronyms = False):
	return caseToken(preSplit(phrase), keepAcronyms)

def splitAndCase(phrase, keepAcronyms = False):
	return map(lambda t: caseToken(t, keepAcronyms), str.split(preSplit(phrase)))

def validateTokens(phrase, keepAcronyms = False, tokenValidator = functools.partial(isValidToken, minLength = 2), phraseValidator = isValidPhrase):
	if phrase:
		tokens = splitAndCase(phrase, keepAcronyms)
		validTokens = []
		for token in tokens:
			if tokenValidator(token): validTokens.append(token)
		if phraseValidator(validTokens): return validTokens
	return []

def normalizeAndValidatePhrase(value,
	keepAcronyms = False, tokenValidator = functools.partial(isValidToken, minLength = 2), phraseValidator = isValidPhrase):
	''' Returns a string that joins normalized, valid tokens for the input phrase
		(None if no valid tokens were found) '''
	tokens = validateTokens(value, keepAcronyms, tokenValidator, phraseValidator)
	return ' '.join(tokens) if len(tokens) > 0 else None

def stripped(s): return s.strip(" -_.,'?!").strip('"').strip()

def fileRowIterator(filePath, sep):
	with open(filePath, mode = 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter = sep, quotechar='"')
		for row in reader:
			try:
				yield list(map(stripped, row))
			except UnicodeDecodeError as ude:
				logging.error('Unicode error while parsing "%s"', row)

def fileToVariantMap(fileName, sep = '|', includeSelf = False):
	''' The input format is pipe-separated, column 1 is the main variant, column 2 an alternative variant.

		Returns a reverse index, namely a map from original alternative variant to original main variant

		Parameters:
		includeSelf if True, then the main variant will be included in the list of alternative variants
			(so as to enable partial matching simultaneously).
	'''
	otherToMain = defaultdict(list)
	mainToOther = defaultdict(list)
	for r in fileRowIterator(fileName, sep):
		if len(r) < 2: continue
		main, alts = r[0], r[1:]
		for alt in alts: otherToMain[alt].append(main)
		mainToOther[main].extend(alts)
	l = list([(other, next(iter(main))) for (other, main) in otherToMain.items() if len(main) < 2])
	if includeSelf: l = list([(main, main) for main in mainToOther.keys()]) + l
	return dict(l)


def fileToList(fileName, path = 'resource'): 
	filePath = fileName if path is None else os.path.join(path, fileName)
	with open(filePath, mode = 'r') as f: 
		return [justCase(line) for line in f]

FRENCH_WORDS = set(fileToList('liste_mots_fr.col'))

STOP_WORDS_FR = set([
	# Prepositions (excepted "avec" and "sans" which are semantically meaningful)
	"a", "au", "aux", "de", "des", "du", "par", "pour", "sur", "chez", "dans", "sous", "vers",
	# Articles
	"le", "la", "les", "l", "c", "ce", "ca",
	 # Conjonctions of coordination
	"mais", "et", "ou", "donc", "or", "ni", "car",
])

STOP_WORDS_EN = set(','.split("a, aboard, about, above, across, after, again, against, all, almost, alone, along, alongside, already, also, although, always, am, amid, amidst, among, amongst, an, and, another, anti, any, anybody, anyone, anything, anywhere, are, area, areas, aren't, around, as, ask, asked, asking, asks, astride, at, aught, away, back, backed, backing, backs, bar, barring, be, became, because, become, becomes, been, before, began, behind, being, beings, below, beneath, beside, besides, best, better, between, beyond, big, both, but, by, came, can, can't, cannot, case, cases, certain, certainly, circa, clear, clearly, come, concerning, considering, could, couldn't, daren't, despite, did, didn't, differ, different, differently, do, does, doesn't, doing, don't, done, down, down, downed, downing, downs, during, each, early, either, end, ended, ending, ends, enough, even, evenly, ever, every, everybody, everyone, everything, everywhere, except, excepting, excluding, face, faces, fact, facts, far, felt, few, fewer, find, finds, first, five, following, for, four, from, full, fully, further, furthered, furthering, furthers, gave, general, generally, get, gets, give, given, gives, go, goes, going, good, goods, got, great, greater, greatest, group, grouped, grouping, groups, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, he's, her, here, here's, hers, herself, high, high, high, higher, highest, him, himself, his, hisself, how, how's, however, i, i'd, i'll, i'm, i've, idem, if, ilk, important, in, including, inside, interest, interested, interesting, interests, into, is, isn't, it, it's, its, itself, just, keep, keeps, kind, knew, know, known, knows, large, largely, last, later, latest, least, less, let, let's, lets, like, likely, long, longer, longest, made, make, making, man, many, may, me, member, members, men, might, mightn't, mine, minus, more, most, mostly, mr, mrs, much, must, mustn't, my, myself, naught, near, necessary, need, needed, needing, needn't, needs, neither, never, new, new, newer, newest, next, no, nobody, non, none, noone, nor, not, nothing, notwithstanding, now, nowhere, number, numbers, of, off, often, old, older, oldest, on, once, one, oneself, only, onto, open, opened, opening, opens, opposite, or, order, ordered, ordering, orders, other, others, otherwise, ought, oughtn't, our, ours, ourself, ourselves, out, outside, over, own, part, parted, parting, parts, past, pending, per, perhaps, place, places, plus, point, pointed, pointing, points, possible, present, presented, presenting, presents, problem, problems, put, puts, quite, rather, really, regarding, right, right, room, rooms, round, said, same, save, saw, say, says, second, seconds, see, seem, seemed, seeming, seems, seen, sees, self, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, show, showed, showing, shows, side, sides, since, small, smaller, smallest, so, some, somebody, someone, something, somewhat, somewhere, state, states, still, still, such, suchlike, sundry, sure, take, taken, than, that, that's, the, thee, their, theirs, them, themselves, then, there, there's, therefore, these, they, they'd, they'll, they're, they've, thine, thing, things, think, thinks, this, those, thou, though, thought, thoughts, three, through, throughout, thus, thyself, till, to, today, together, too, took, tother, toward, towards, turn, turned, turning, turns, twain, two, under, underneath, unless, unlike, until, up, upon, us, use, used, uses, various, versus, very, via, vis-a-vis, want, wanted, wanting, wants, was, wasn't, way, ways, we, we'd, we'll, we're, we've, well, wells, went, were, weren't, what, what's, whatall, whatever, whatsoever, when, when's, where, where's, whereas, wherewith, wherewithal, whether, which, whichever, whichsoever, while, who, who's, whoever, whole, whom, whomever, whomso, whomsoever, whose, whosoever, why, why's, will, with, within, without, won't, work, worked, working, works, worth, would, wouldn't, ye, year, years, yet, yon, yonder, you, you'd, you'll, you're, you've, you-all, young, younger, youngest, your, yours, yourself, yourselves"))

STOP_WORDS = STOP_WORDS_FR | STOP_WORDS_EN

NON_DISCRIMINATING_TOKENS = list([justCase(t) for t in [
	# FR
	'Société', 'Université', 'Unité', 'Pôle', 'Groupe', 'SA', 'Entreprise',
	# EN
	'Society', 'University', 'Hospital', 'Department', 'Group', 'Ltd'
	'Agency', 'Institute', 'College', 'Faculty', 'Authority', 
	'Academy', 'Department', 'Center', 'Centre', 'School',  'Enterprise', 'Company',
	'Foundation', 'City', 'Clinic', 'Consulting', 'Organization',
	# DE
	'Klinikum', 'Hochschule', 'Fachhochschule',
	# IT
	'Istituto', 'Regione', 'Comune', 'Centro',
	# ES
	'Universidad', 'Agencia', 'Servicio', 'Conselleria', 'Museo', 'Fundacion',
	# PL
	'Uniwersytet', 'Centrum', 'Akademia'
]])

def inverse_translation_map(variants_by_main):
	inverse_map = dict()
	for main, variants in variants_by_main.items():
		main0 = justCase(main)
		for variant in variants:
			inverse_map[justCase(variant)] = main0
	return inverse_map

TRANSLATIONS = inverse_translation_map({
	'University': ['Université', 'Universidad', 'Universität', 'Universitat', 'Univ', 'Universita'],
	'Laboratory': ['Lab', 'Laboratoire', 'Labo'],
	'Hospital': ['Hôpital'],
	'Agency': ['Agence', 'Agencia'],
	'Department': ['Dipartimento', 'Département', 'Dpto', 'Dpt'],
	'City': ['Commune', 'Comune'],
	'Clinic': ['Clinique', 'Klinikum'],
	'CH': ['Complejo Hospitalario', 'Centre Hospitalier'],
	'Academy': ['Académie', 'Akademia', 'Aca'],
	'Institute': ['Institut', 'Instituto', 'Istituto', 'Instytut'],
	'Center': ['Centre', 'Centrum', 'Zentrum'],
	'Association': ['Asociacion'],
	'Society': ['Société', 'Societa', 'Gesellschaft'],
	'Development': ['Développement'],
	'Consulting': ['Conseil'],
	'Foundation': ['Fundacion', 'Fondation'],
	'European': ['Européen'],
	'Technology': ['Technologie'],
	'Systems': ['Systèmes'],
	'School': ['École', 'Escuela', 'Scuola'],
	'Industrial': ['Industriel', 'Industrie', 'Industrial'],
	'Research': ['Recherche'],
	'UM': ['unité mixte'],
	'Medical Center': ['MC'],
	'Energy': ['Energie', 'Energia', 'Power'],
	'Organization': ['Ograniczona'],
	'Institute': ['Inst', 'Institut', 'Institució', 'Institucion'],
	'Technical University': ['TU', 'Technische Universität', 'Technical Univ', 'Tech Univ'],
	'Limited': ['Ltd']
})

def translate(phrase):
	s = justCase(phrase)
	for (source, target) in TRANSLATIONS.items():
		s = re.sub(r"\b%s\b" % source, target, s)
	return s

def inverse_regex_map(regexes_by_main):
	inverse_map = dict()
	for main, regexes in regexes_by_main.items():
		main0 = justCase(main)
		for regex in regexes:
			inverse_map[regex.replace('*', '[a-z]*')] = main0
	return inverse_map

CATEGORIES = inverse_regex_map({
	'University': ['univ*', 'facult.', 'campus', 'departe?ment'],
	'School': ['scol*', 'school'],
	'Company': ['ltd', 'inc', 'sas', 'gmbh', 'sarl', 'sa', 'ab'],
	'Medical': ['medic*', 'hospi*', 'hopi.*', 'clini*', 'chu', 'ch', 'klinik', 'service'],
	'Research': ['unite', 'unit', 'lab*', 'recherche', 'umr', 'ufr', 'cnrs', 'cea'],
	'Other': ['committee', 'comite', 'agence', 'institute', 'bureau']
})

def categorize(token):
	s = justCase(token)
	for (regex, category) in CATEGORIES.items():
		if re.findall(regex, s, re.I):
			return category
	return None

def makeKey(country, city): 
	return caseToken(country) if country else ''

def countryToCodeMap():
	with open('resource/country_name_code.csv') as f:
		reader = csv.reader(f, delimiter = ',')
		return dict([(row[0], 'UK' if row[1] == 'GB' else row[1]) for row in reader])

def lowerOrNot(token, keepAcronyms = False, keepInitialized = False):
	''' Set keepAcronyms to true in order to improve precision (e.g. a CAT scan will not be matched by a kitty). '''
	if keepAcronyms and len(token) >= MIN_ACRO_SIZE and len(token) <= MAX_ACRO_SIZE and isAcroToken(token):
		return token
	if keepInitialized:
		m = re.search("([A-Z][0-9]+)[^'a-zA-Z].*", token)
		if m:
			toKeep = m.group(0)
			return toKeep + lowerOrNot(token[len(toKeep):], keepAcronyms, keepInitialized)
	return token.lower()

# A map from alt variant to main variant 
SYNMAP = fileToVariantMap('resource/grid_synonyms')

def normalize(t): 
	s = caseToken(t)
	for variant, main in SYNMAP.items():
		s = s.replace(variant, main)
	return s

def filterProperNouns(tokens):
	return list([t for t in tokens if len(t) > 2 and t not in FRENCH_WORDS  and (DONT_DISCRIMINATE or t not in NON_DISCRIMINATING_TOKENS) and not any([d.check(t) for d in DICTS])])
	
def score_chars(src, ref):
	# Returns a score in [0, 100]
	a0 = toASCII(src)
	b0 = toASCII(ref)
	a1 = acronymizePhrase(a0)
	b1 = acronymizePhrase(b0)
	if len(a1) > 0 and len(b1) > 0 and (a1 == b0.upper() or a0.upper() == b1):
		logging.debug('Accepted for ACRO : {} / {}'.format(a, b))
		return 100
	a = justCase(src)
	b = justCase(ref)
	absCharRatio = fuzz.ratio(a, b)
	if absCharRatio < 20: 
		logging.debug('Rejected for ABS : {} / {}'.format(a, b))
		return 0
	partialCharRatio = fuzz.partial_ratio(a, b)
	if partialCharRatio < 30: 
		logging.debug('Rejected for PARTIAL : {} / {}'.format(a, b))
		return 0
	return absCharRatio * partialCharRatio / 100

def score_tokens(src, ref, translate_tokens):
	if translate_tokens:
		return score_tokens(translate(src), translate(ref), False)
	# Returns a score in [0, 100]
	aTokens = validateTokens(src)
	bTokens = validateTokens(ref)
	a2 = ' '.join(aTokens)
	b2 = ' '.join(bTokens)
	tokenSortRatio = fuzz.token_sort_ratio(a2, b2)
	if tokenSortRatio < 40: 
		logging.debug('Rejected for TOKEN_SORT : {} / {}'.format(src, ref))
		return 0
	tokenSetRatio = fuzz.token_set_ratio(a2, b2)
	if tokenSetRatio < 50:
		logging.debug('Rejected for TOKEN_SET : {} / {}'.format(src, ref))
		return 0
	if REQUIRES_SHARED_PROPER_NOUN:
		aProper = ' '.join(filterProperNouns(aTokens))
		bProper = ' '.join(filterProperNouns(bTokens))
		# if(len(aProper) > 3 and len(bProper) > 3):
		if len(aProper) > 0 or len(bProper) > 0:
			properNounSortRatio = fuzz.token_sort_ratio(aProper, bProper)
			if properNounSortRatio < 80: 
				logging.debug('Rejected for PROPER_NOUN_SORT : {} / {}'.format(src, ref))
				return 0
			properNounSetRatio = fuzz.token_set_ratio(aProper, bProper)
			if properNounSetRatio < 60:
				logging.debug('Rejected for PROPER_NOUN_SET : {} / {}'.format(src, ref))
				return 0
	return tokenSortRatio * tokenSetRatio / 100

def scoreStrings(src, ref):
	s =  max(score_chars(src, ref), score_tokens(src, ref, False), score_tokens(src, ref, True))
	return s if s > 60 else 0

def score_items(src, ref):
	if len(src['categories']) > 0 and len(ref['categories']) > 0 and len(src['categories'] & ref['categories']) < 1:
		return (0, None)
	src_label_variants = set([src['label']]) | src['variants']
	if 'acronym' in src: 
		src_label_variants.add(src['acronym'])
	if EXCLUDE_ADDRESS_AS_LABEL and 'address_as_label' in src: 
		src_label_variants.discard(src['address_as_label'])
	ref_label_variants = set([ref['label']]) | ref['variants'] | ref['aliases']
	score_variants = (0, None)
	if len(src_label_variants) > 0 and len(ref_label_variants) > 0:
		max_score_pair = None
		for a in src_label_variants:
			for b in ref_label_variants:
				score = scoreStrings(a, b)
				if max_score_pair is None or max_score_pair[0] < score:
					max_score_pair = (score, a, b)
		score_variants = (max_score_pair[0], 'Matching variants "{}" / "{}"'.format(max_score_pair[1], max_score_pair[2]))
	score_acro = (0, None)
	if 'acronym' in ref:
		if ref['acronym'] in src_label_variants:
			score_acro = (100, 'Matching acronym "{}" / label "{}"'.format(ref['acronym'], src['label']))
	elif 'acro' in src and 'acro' in ref:
		max_score_pair = None
		for a in src['acro']:
			for b in ref['acro']:
				score = fuzz.ratio(a, b)
				if max_score_pair is None or max_score_pair[0] < score:
					max_score_pair = (score, a, b)
		if max_score_pair[0] > 0:
			score_acro = (max_score_pair[0], 'Matching acronyms "{}" / "{}"'.format(max_score_pair[1], max_score_pair[2]))
	score_country = (50, None)
	if 'country' in src and 'country' in ref:
		if fuzz.ratio(src['country'], ref['country']) > 80:
			score_country = (100, 'Matching countries {} / {}'.format(src['country'], ref['country']))
		else:
			score_country = (0, None)
	score_city = (50, None)
	if 'city' in src and 'city' in ref:
		if fuzz.ratio(src['city'], ref['city']) > 80:
			score_city = (100, 'Matching cities {} / {}'.format(src['city'], ref['city']))
		else:
			score_city = (0, None)
	score_ur = (0, None)
	if 'ur_id' in src and 'ur_id' in ref and src['ur_id'] == ref['ur_id']:
		score_ur = (100, 'Matching research unit IDs {} / {} '.format(src['ur_id'], ref['ur_id']))
	score_url = (0, None)
	if 'url' in ref:
		link = ref['url']
		i = link.find('.')
		if i > 0:
			j = link[i+1:].find('.')
			if j > 0:
				domain_name = justCase(link[i+1:j].replace('-', ' '))
				if domain_name in list([justCase(s) for s in src_label_variants]):
					score_url = (100, 'Matching URL "{}" / label "{}"'.format(domain_name, src['label']))
	score_str = sorted([score_variants, score_acro, score_ur, score_url], key = lambda p: p[0], reverse = True)[0]
	item_score = score_str[0] * score_country[0] * score_city[0]
	reasons = [score_str[1]]
	if score_country[1] is not None: reasons.append(score_country[1])
	if score_city[1] is not None: reasons.append(score_city[1])
	return (item_score  / 100**2, ' + '.join(reasons)) if score_str[0] >= MIN_STRING_SCORE and item_score >= (MIN_STRING_SCORE * 50 * 100) else (0, '')

def gridded_count(src_items_by_label):
	return sum(['grid' in src_item or 'parent_grid' in src_item for src_item in src_items_by_label.values()])

COUNTRY_SINS = dict()
REF_ITEM_BY_GRID = dict()
GRIDS_BY_TOKEN = defaultdict(set)
ALL_CITIES = set()
ACRONYM_COUNT = Counter()
with open('resource/country_fr_en.csv') as countryFile:
	country_reader =  csv.DictReader(countryFile, delimiter = '|')
	for country_row in country_reader:
		country_fr = country_row['Pays']
		country_en = country_row['Country']
		COUNTRY_SINS[justCase(country_fr)] = country_en
		COUNTRY_SINS[justCase(country_en)] = country_en
with open('resource/grid.csv') as refFile:
	ref_reader =  csv.DictReader(refFile, delimiter = ',', quotechar = '"')
	for refRow in ref_reader:
		label = refRow['Name']
		tokens = validateTokens(label)
		grid = refRow['ID']
		country = refRow['Country']
		city = refRow['City']
		if exclude_FR and country == 'France': continue
		ref_item = dict( 
				origin = REFERENCE, country = country, city = city, label = label, tokens = tokens, labels = dict(), categories = set(), 
				grid = grid, variants = set([label]), acros = set(), aliases = set(), children = set() )
		if len(country) > 0:
			ref_item['variants'].add(' '.join([label, country]))
			if len(city) > 0:
				ALL_CITIES.add(justCase(city))
			if len(country) > 0:
				ref_item['variants'].add(' '.join([label, city, country]))
		state = refRow['State']
		if len(city) > 0 and len(country) > 0 and len(state) > 0:
			ref_item['variants'].add(' '.join([label, city, state, country]))
		enrich_item_with_variants(ref_item)
		REF_ITEM_BY_GRID[grid] = ref_item
		for token in tokens: 
			GRIDS_BY_TOKEN[token].add(grid)
		if len(REF_ITEM_BY_GRID) % 1000 == 0: logging.warning('Pre-processed {} reference entries'.format(len(REF_ITEM_BY_GRID)))
with open('resource/grid_aliases.csv') as alias_file:
	aliases_reader = csv.DictReader(alias_file, delimiter = ',', quotechar = '"')
	for alias_row in aliases_reader:
		grid = alias_row['grid_id']
		if grid not in REF_ITEM_BY_GRID: continue
		REF_ITEM_BY_GRID[grid]['aliases'].add(alias_row['alias'])
with open('resource/grid_labels.csv') as labels_file:
	labels_reader = csv.DictReader(labels_file, delimiter = ',', quotechar = '"')
	for label_row in labels_reader:
		grid = label_row['grid_id']
		if grid not in REF_ITEM_BY_GRID: continue
		REF_ITEM_BY_GRID[grid]['labels'][label_row['iso639']] = label_row['label']
with open('resource/grid_acronyms.csv') as acronyms_file:
	acronyms_reader = csv.DictReader(acronyms_file, delimiter = ',', quotechar = '"')
	for acronym_row in acronyms_reader:
		grid = acronym_row['grid_id']
		if grid not in REF_ITEM_BY_GRID: continue
		REF_ITEM_BY_GRID[grid]['acronym'] = acronym_row['acronym']
		ACRONYM_COUNT[acronym_row['acronym']] += 1
with open('resource/grid_links.csv') as links_file:
	links_reader = csv.DictReader(links_file, delimiter = ',', quotechar = '"')
	for link_row in links_reader:
		grid = link_row['grid_id']
		if grid not in REF_ITEM_BY_GRID: continue
		REF_ITEM_BY_GRID[grid]['url'] = link_row['link']
with open('resource/grid_relationships.csv') as rels_file:
	rels_reader = csv.DictReader(rels_file, delimiter = ',', quotechar = '"')
	for rel_row in rels_reader:
		grid = rel_row['grid_id']
		rel_grid = rel_row['related_grid_id']
		if grid not in REF_ITEM_BY_GRID or rel_grid not in REF_ITEM_BY_GRID: continue
		if rel_row['relationship_type'] == 'Child':
			REF_ITEM_BY_GRID[grid]['children'].add(rel_grid)
			REF_ITEM_BY_GRID[rel_grid]['parent'] = grid
		elif rel_row['relationship_type'] == 'Parent':
			REF_ITEM_BY_GRID[rel_grid]['children'].add(grid)
			REF_ITEM_BY_GRID[grid]['parent'] = rel_grid

def grid_label_set(labels):
	src_items_by_label = dict()
	token_count = Counter()
	for doc_id, label in enumerate(labels):
		tokens = validateTokens(label)
		cities = set([justCase(t) for t in tokens]) | ALL_CITIES
		countries = list()
		country_variant = list()
		for token in tokens:
			token_count[token] += 1
			if token in COUNTRY_SYNS: 
				countries.append(COUNTRY_SYNS[token])
			else:
				country_variant.append(token)
		src_item = dict(doc_id = doc_id, origin = SOURCE, label = label, categories = set(), tokens = tokens, variants = set([label]), acros = set())
		if len(cities) == 1:
			src_item['city'] = cities[0]
		if len(countries) > 0: 
			src_item['country'] = countries[0]
			if ADD_VARIANT_WITHOUT_COUNTRY:
				src_item['variants'].add(' '.join(country_variant))
		enrich_item_with_variants(src_item)
		src_items_by_label[label] = src_item
	unmatched_src_labels = set()
	for (src_label, src_item) in src_items_by_label.items():
		candidate_grids = set()
		for token in sorted(src_item['tokens'], key = lambda t: token_count[t], reverse = True)[:8]:
			newGrids = candidate_grids | GRIDS_BY_TOKEN[token]
			if len(newGrids) > 32: break
			candidate_grids = newGrids
		logging.debug('Source label "{}": found {} candidates'.format(item_to_str(src_item), len(candidate_grids)))
		if len(candidate_grids) < 1: 
			continue
		candidate = sorted([(grid, score_items(src_item, REF_ITEM_BY_GRID[grid])) for grid in candidate_grids], key = lambda p: p[1][0], reverse = True)[0]
		score = candidate[1][0]
		if score > 0: 
			grid = candidate[0]
			reason = candidate[1][1]
			ref_label = REF_ITEM_BY_GRID[grid]['label']
			logging.info('Match found for src item "{}": "{}" ({}) <-- {}'.format(item_to_str(src_item), ref_label, score, reason))
			src_item['grid'] = grid
			src_item['grid_reason'] = reason
		else:
			logging.info('No match found for src item "{}"'.format(item_to_str(src_item)))
			unmatched_src_labels.add(src_label)
	logging.warning('==> now {} gridded entities'.format(gridded_count(src_items_by_label)))
	if FORBID_DUPE_ACRONYMS:
		for src_item in src_items_by_label.values():
			if 'acronym' in src_item and ACRONYM_COUNT[src_item['acronym']] > 1: del src_item['acronym']
		for ref_item in REF_ITEM_BY_GRID.values():
			if 'acronym' in ref_item and ACRONYM_COUNT[ref_item['acronym']] > 1: del ref_item['acronym']
	logging.warning('Matching unmatched src labels...')
	attached_parent_grid_count = 0
	src_items_index = dict()
	for (src_label, src_item) in src_items_by_label.items():
		src_items_index[src_label] = src_item
		if 'acronym' in src_item and (FORBID_DUPE_ACRONYMS or ACRONYM_COUNT[src_item['acronym']] < 2): src_items_index[src_item['acronym']] = src_item
	for src_label in unmatched_src_labels:
		src_item = src_items_by_label[src_label]
		if 'parent_label' not in src_item: continue
		parent_label = src_item['parent_label']
		if parent_label in src_items_index:
			if grid in src_items_index[parent_label]:
				logging.info('Attaching grid of matched parent "{}" to child "{}"'.format(parent_label, item_to_str(src_item)))
				src_item['grid'] = src_items_index[parent_label]['grid']
				attached_parent_grid_count += 1
			else:
				logging.info('Unmatched parent "{}" of child "{}"'.format(parent_label, item_to_str(src_item)))
		else:
			logging.warning('Parent "{}" of child "{}" not found!'.format(parent_label, item_to_str(src_item)))
	logging.warning('Matched {} unmatched src labels  ==> now {} gridded entities'.format(attached_parent_grid_count, gridded_count(src_items_by_label)))
	logging.warning('Checking parent-child relationships...')
	added_parent_grid_count = 0
	for (src_label, src_item) in src_items_by_label.items():
		if 'grid' in src_item:
			ref_item = REF_ITEM_BY_GRID[src_item['grid']]
			if 'parent' in ref_item:
				parent_item = REF_ITEM_BY_GRID[ref_item['parent']]
				logging.info('Adding "{}" parent grid of "{}"'.format(parent_item['label'], item_to_str(src_item)))
				src_item['parent_grid'] = parent_item['grid']
				added_parent_grid_count += 1
	logging.warning('Added {} parent grids ==> now {} gridded entities'.format(added_parent_grid_count, gridded_count(src_items_by_label)))
	logging.warning('Post-processing gridded entries...')
	for (src_label, src_item) in src_items_by_label.items():
		ref_item = REF_ITEM_BY_GRID[src_item['grid']] if 'grid' in src_item else REF_ITEM_BY_GRID[src_item['parent_grid']] if 'parent_grid' in src_item else None
		if ref_item is not None:
			src_item['city'] = ref_item['city']
			src_item['country'] = ref_item['country']
			src_item['grid_label'] = ref_item['label']
	return src_items_by_label
