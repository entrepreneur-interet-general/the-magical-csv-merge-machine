# Data preprocessing

This section describes the main preprocessing steps applied to source and referential data prior, either prior to a linking operation between a source file and a referential, or as a normalization project by itself. 

__Data type inference__

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

__Data type normalization__

Following type inference, both automatic and manual (by the overriding mechanism described above), the code tries to normalize each value in a column based on its associated type, in order to more coherently, correctly and consistently recode those values, as well as to facilitate the linking operation.

As a basic example of how it helps during linking: normalization can in some cases replace a term or an entire phrase by a variant (such as a synonym, or replacing an acronym by its expanding form) which will make purely string-based fuzzy matching more feasible - or merely feasible in this case.

The codebase contains different types of normalization methods, with a n-n mapping between normalizers and data types. 