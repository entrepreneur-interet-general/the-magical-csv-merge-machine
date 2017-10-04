# Resource maintenance

This section describes how resource files used by the pre-processing stages can be maintained and updated throughout the application's lifetime. Those resource files differ from referentials, in that their data is built into the application and not exposed to the end user.

__Resource file types__

Resource files are located in the `resource/` folder below the project root and include:

* _Label lists_ (without an extension), for example `resource/departement` contains the list of all official, normalized "département" names in France. In case an administrative redistricting occurs (as was the case in 2017), this resource file needs to be updated. Other examples of label lists are `resource/prenom` (a non-exhaustive list of first names commonly used in France), `resource/titre_revue` (a non-exhaustive list of periodical publications), etc. Thus label lists are used both for type inference (i.e. a field whose values often - exactly or closely - match the label list are assigned the corresponding type) and for normalization (i.e. when there is a match, even a fuzzy one, the canonical variant found in the label list is picked).

* _Vocabulary lists_ (with the extension `.vocab`), for example `resource/etab_enssup.vocab` which contains frequently occurring, discriminating terms (tokens or phrases). Thus vocabulary lists are used for type inference only (i.e. a field whose values often contain a certain amount of vocabulary based on criteria such as token count, token ratio, char ratio, etc. will be assigned the corresponding type) and not for value normalization.

* _Synonym lists_ (with the extension `.syn`), for example `resource/org_entreprise.syn` which contains a number of generic denominations for companies in France, and when applicable variations over those denominations (e.g. `Société coopérative ouvrière de production` and its acronym `SCOP` are synonyms). Thus synonym lists are used both for type inference (i.e. mapping of each secondary variant to a main variant ensures more coverage in the type inference hits) and for value normalization (since all variants are converted to the main one).

* _Column values_ (with the extension `.col`), for example `resource/org_rnsr.col`. Such files contain a list of values extracted from an actual data file (as opposed to label lists which are either manually constructed or fetched from an official nomenclature, but in any case are somehow curated). They can be used as substitutes for label list files when no such curated list exists (as is the case for `org_rnsr.col` by definition), and are used both for type inference and normalization, although of course care must be taken to make column value lists as exhaustive as possible when using them for normalization.

__Maintenance process__

To enable anyone to push changes to the resources, even a non-developer, the process just consists of replacing any number of the files described above in the git repository (all those files are independent from one another).
