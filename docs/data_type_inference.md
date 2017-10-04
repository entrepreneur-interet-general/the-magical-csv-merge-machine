# Data type inference

## Description

For each column in a CSV file, the code attempts to associate a likelihood that the values are of a given type (in a loose sense, meaning that if _most_ values match the type and/or the matching is only partial i.e. on a substring of the value strings, then the column might still be attributed that data type).

The set of all supported data types is returned by some API calls, for example in the `all_types` attribute of the `recode_types` verb. Here is a current snapshot of that attribute, showing that the types are organized in a flat hierarchy, wherein each type has zero or one parent type.

```json
"all_types" : {
      "ID" : [
         "ID personne",
         "ID publication",
         "ID organisation"
      ],
      "Entité agro" : [
         "Phyto"
      ],
      "Adresse" : [
         "Commune",
         "Voie",
         "Code Postal",
         "Pays"
      ],
      "Publication" : [
         "Résumé",
         "ID publication",
         "Article",
         "Titre de revue"
      ],
      "Entité biomédicale" : [
         "Spécialité médicale",
         "Nom d'essai clinique"
      ],
      "Entité Géo" : [
         "Commune",
         "Code Postal",
         "Adresse",
         "Département",
         "Pays",
         "Région"
      ],
      "Nom de personne" : [
         "Prénom",
         "Titre"
      ],
      "Institution de recherche" : [
         "Collaborateur d'essai clinique",
         "Structure de recherche",
         "Partenaire de recherche"
      ],
      "Article" : [
         "Résumé",
         "ID publication",
         "Contenu d'article"
      ],
      "Date" : [
         "Année"
      ],
      "Etablissement" : [
         "Etablissement d'Enseignement Supérieur"
      ],
      "ID personne" : [
         "Nom de personne",
         "Email",
         "Téléphone",
         "NIR"
      ],
      "Type structuré" : [
         "Date",
         "Email",
         "Téléphone",
         "URL"
      ],
      "Institution" : [
         "Structure de recherche",
         "Entreprise",
         "Etablissement"
      ],
      "Texte" : [
         "Anglais",
         "Résumé",
         "Article",
         "Franéais"
      ],
      "Entité MESR" : [
         "Domaine de Recherche",
         "Académie",
         "Etablissement",
         "Mention APB",
         "Institution de recherche"
      ],
      "ID publication" : [
         "ISSN",
         "DOI"
      ],
      "Autres types" : [
         "Entité agro",
         "Institution",
         "Entité Géo",
         "Publication",
         "Acronyme",
         "Type structuré",
         "Entité biomédicale",
         "TVA",
         "Texte",
         "NIF",
         "Education Nationale",
         "ID",
         "Entité MESR",
         "Nom"
      ],
      "ID organisation" : [
         "SIREN",
         "Numéro National de Structure",
         "SIRET",
         "UAI",
         "Numéro UMR"
      ]
   }
```

After the system associates zero or one likeliest type per column, the user can confirm or infirm each inferred type, as well as specify a type for those columns that were missed during inference. 

## Implementation

Type inference relies on a number of classes that try to match each cell value against a type, using one of the following methods:

__String matching__

- `LabelMatcher` that does exact or fuzzy matching against either the entire cell value or a part of it. It can be parameterized by a lexicon (containing reference labels), the match mode (exact of approximate), and a dictionary of synonyms (to provide a minimal amount of normalization in what is otherwise very rudimentary matching).
- `RegexMatcher` uses a regex to match part or all of the cell value. Useful for identifier types.
- `TokenizedMatcher` applies tokenization to both a set of reference string called "lexicon" and the input cell values, and matches based on the number of matched tokens. It can be parameterized by a custom scoring function, a max token count (in case some strings are very long, and this is done via a sliding window), the min number of distinct shared tokens, and a list of stopwords.
- `StdnumMatcher` performs regex-based matching for a number of official identifier types:
   - NIR, NIF, TVA as documented by [https://fr.wikipedia.org/wiki/Code_Insee#Identification_des_individus]
   - SIREN, SIRET

__Custom matching__

- `GridMatcher` is even more custom than the others, since it applies to entities that can be associated to one or several GRID identifiers. However it is only a partial port of an ad-hoc grid-matching project which leveraged a number of fields in addition to the sole entity label (like geolocation, parent label, etc.), which is impossible here since the framework only allows per-column normalization, independently from the other columns in the input CSV 
- `CustomDateMatcher` relies the dateparser library to take best guesses at dates, in particular whether they are written in the 'DDMM...' form (day of the month first, as in Western Europe) or in the 'MMDD...' form (month first, as in the US), based on relative frequency
- `CustomTelephoneMatcher` uses the phonenumbers library (https://pypi.python.org/pypi/phonenumbers) to match phone numbers in most countries and locales
- `CustomPersonNameMatcher` uses ad-hoc code to parse the most common constructs used for writing French person names
- `CustomAddressMatcher` uses the libpostal library and works particularly well for US addresses
- `FrenchAddressMatcher` uses the BAN web service (http://api-adresse.data.gouv.fr) and works well for French addresses

__Compound matchers__

- `CompositeMatcher` matches a type when a number of type components are matched. For example, an address match occurs when there is at least a match on the street type and the city or zip type.
- `SubtypeMatcher` matches a type when any of its subtypes is matched. For example, an "établissement" match occurs when there is either an "établissement d'enseignement supérieur" match or an "établissement du 1er et 2nd degré" match.
- `CompositeRegexMatcher` is used when several children fields of a parent composite field can be captured as groups within the same regex. For example, a pattern of the form "last_name, first_name" (with a literal comma separating both components) will be defined to match a person name.

__Variational matchers__

- `AcronymMatcher` computes acronyms on-the-fly from input cell values. Like in the case of `GridMatcher`, this is a degraded (simplified) version of more general acronym matching code which starts by fetching common acronyms from reference data (thus operating a kind of "training" on the acronym matching algorithm) then disambiguating those acronyms using common nouns and the like, and finally matching them against input values based on frequency information. Here acronyms are only constructed the reference and input values, and no frequency or stopword information is used. The only parameters in this version are min and max acronym lengths.
- `VariantExpander` maps variants of a term to its main variant, thus operating a pre-normalization of cell values prior to type matching. An instance of this class usually takes another `TypeMatcher` in its constructor, which is then handed off the cell values after all of their secondary variants are expanded into their main variant.
