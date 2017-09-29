# Data preprocessing

This section describes the main preprocessing steps applied to source and referential data prior, either prior to a linking operation between a source file and a referential, or as a normalization project by itself. 

__Data type inference__

_Description_

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

_Implementation_

Type inference relies on a number of classes that try to match each cell value against a type, using one of the following methods:
- String matching
   - `LabelMatcher` that does exact or fuzzy matching against either the entire cell value or a part of it. It can be parameterized by a lexicon (containing reference labels), the match mode (exact of approximate), and a dictionary of synonyms (to provide a minimal amount of normalization in what is otherwise very rudimentary matching).
   - `RegexMatcher` uses a regex to match part or all of the cell value. Useful for identifier types.
   - `TokenizedMatcher` applies tokenization to both a set of reference string called "lexicon" and the input cell values, and matches based on the number of matched tokens. It can be parameterized by a custom scoring function, a max token count (in case some strings are very long, and this is done via a sliding window), the min number of distinct shared tokens, and a list of stopwords.
   - `StdnumMatcher`
- Custom matching:
   - `GridMatcher` is even more custom than the others, since it applies to entities that can be associated to one or several GRID identifiers. However it is only a partial port of an ad-hoc grid-matching project which leveraged a number of fields in addition to the sole entity label (like geolocation, parent label, etc.), which is impossible here since the framework only allows per-column normalization, independently from the other columns in the input CSV 
   - `CustomDateMatcher` relies the dateparser library to take best guesses at dates, in particular whether they are written in the 'DDMM...' form (day of the month first, as in Western Europe) or in the 'MMDD...' form (month first, as in the US), based on relative frequency
   - `CustomTelephoneMatcher` uses the phonenumbers library (https://pypi.python.org/pypi/phonenumbers) to match phone numbers in most countries and locales
   - `CustomPersonNameMatcher` uses ad-hoc code to parse the most common constructs used for writing French person names
   - `CustomAddressMatcher` uses the libpostal library and works particularly well for US addresses
   - `FrenchAddressMatcher` uses the BAN web service (http://api-adresse.data.gouv.fr) and works well for French addresses
- Compound matchers
   - `CompositeMatcher` matches a type when a number of type components are matched. For example, an address match occurs when there is at least a match on the street type and the city or zip type.
   - `SubtypeMatcher` matches a type when any of its subtypes is matched. For example, an "établissement" match occurs when there is either an "établissement d'enseignement supérieur" match or an "établissement du 1er et 2nd degré" match.
   - `CompositeRegexMatcher` is used when several children fields of a parent composite field can be captured as groups within the same regex. For example, a pattern of the form "last_name, first_name" (with a literal comma separating both components) will be defined to match a person name.
- Variational matchers
   - `AcronymMatcher` computes acronyms on-the-fly from input cell values. Like in the case of `GridMatcher`, this is a degraded (simplified) version of more general acronym matching code which starts by fetching common acronyms from reference data (thus operating a kind of "training" on the acronym matching algorithm) then disambiguating those acronyms using common nouns and the like, and finally matching them against input values based on frequency information. Here acronyms are only constructed the reference and input values, and no frequency or stopword information is used. The only parameters in this version are min and max acronym lengths.
   - `VariantExpander` maps variants of a term to its main variant, thus operating a pre-normalization of cell values prior to type matching. An instance of this class usually takes another `TypeMatcher` in its constructor, which is then handed off the cell values after all of their secondary variants are expanded into their main variant.

__Data type normalization__

Following type inference, both automatic and manual (by the overriding mechanism described above), the code tries to normalize each value in a column based on its associated type, in order to more coherently, correctly and consistently recode those values, as well as to facilitate the linking operation.

As a basic example of how it helps during linking: normalization can in some cases replace a term or an entire phrase by a variant (such as a synonym, or replacing an acronym by its expanding form) which will make purely string-based fuzzy matching more feasible - or merely feasible in this case.

The codebase contains different types of normalization methods, with an `n-n` mapping between normalizers and data types. 

Most normalization code belongs to the corresponding `TypeMatcher` class since both phases often have code in common : for example, in order to determine whether a cell value is a person name, it needs to be parsed as a combination of first and last name ; if (based on other cell values' likelihood of being of that type or a different type) the cell's parent column is actually determined to be of the person name type, then normalization just consists in rewriting it as `first_name` followed by a space followed by `last_name`, so the bulk of the work has already been done by the `CustomPersonNameMatcher`!

__Resource maintenance__

This section describes how resource files used by the pre-processing stages can be maintained and updated throughout the application's lifetime. Those resource files differ from referentials, in that their data is built into the application and not exposed to the end user.

_Resource file types_

Resource files are located in the `resource/` folder below the project root and include:

* _Label lists_ (without an extension), for example `resource/departement` contains the list of all official, normalized "département" names in France. In case an administrative redistricting occurs (as was the case in 2017), this resource file needs to be updated. Other examples of label lists are `resource/prenom` (a non-exhaustive list of first names commonly used in France), `resource/titre_revue` (a non-exhaustive list of periodical publications), etc. Thus label lists are used both for type inference (i.e. a field whose values often - exactly or closely - match the label list are assigned the corresponding type) and for normalization (i.e. when there is a match, even a fuzzy one, the canonical variant found in the label list is picked).

* _Vocabulary lists_ (with the extension `.vocab`), for example `resource/etab_enssup.vocab` which contains frequently occurring, discriminating terms (tokens or phrases). Thus vocabulary lists are used for type inference only (i.e. a field whose values often contain a certain amount of vocabulary based on criteria such as token count, token ratio, char ratio, etc. will be assigned the corresponding type) and not for value normalization.

* _Synonym lists_ (with the extension `.syn`), for example `resource/org_entreprise.syn` which contains a number of generic denominations for companies in France, and when applicable variations over those denominations (e.g. `Société coopérative ouvrière de production` and its acronym `SCOP` are synonyms). Thus synonym lists are used both for type inference (i.e. mapping of each secondary variant to a main variant ensures more coverage in the type inference hits) and for value normalization (since all variants are converted to the main one).

* _Column values_ (with the extension `.col`), for example `resource/org_rnsr.col`. Such files contain a list of values extracted from an actual data file (as opposed to label lists which are either manually constructed or fetched from an official nomenclature, but in any case are somehow curated). They can be used as substitutes for label list files when no such curated list exists (as is the case for `org_rnsr.col` by definition), and are used both for type inference and normalization, although of course care must be taken to make column value lists as exhaustive as possible when using them for normalization.

_Maintenance process_

To enable anyone to push changes to the resources, even a non-developer, the process just consists of replacing any number of the files described above in the git repository (all those files are independent from one another).
