# Data type normalization

Following type inference, both automatic and manual (by the overriding mechanism described above), the code tries to normalize each value in a column based on its associated type, in order to more coherently, correctly and consistently recode those values, as well as to facilitate the linking operation.

As a basic example of how it helps during linking: normalization can in some cases replace a term or an entire phrase by a variant (such as a synonym, or replacing an acronym by its expanding form) which will make purely string-based fuzzy matching more feasible - or merely feasible in this case.

The codebase contains different types of normalization methods, with an `n-n` mapping between normalizers and data types. 

Most normalization code belongs to the corresponding `TypeMatcher` class since both phases often have code in common : for example, in order to determine whether a cell value is a person name, it needs to be parsed as a combination of first and last name ; if (based on other cell values' likelihood of being of that type or a different type) the cell's parent column is actually determined to be of the person name type, then normalization just consists in rewriting it as `first_name` followed by a space followed by `last_name`, so the bulk of the work has already been done by the `CustomPersonNameMatcher`!
