# Pseudo API request for the main recoding service

```
request_json = {
                # What data to use / can be combined with passing file through POST
                data:{
                      project_id:ABC, 
                      # Use source from given project, as was computed in module (or by default, in last module)
                      source: {name: my_source, (module:example_module)}, 
                      ref: {name: sirene, internal:True}
                      }
                
                # Module params (indicates what modules to use, with what parameters) 
                # (infer:True will use our own suggestions / not recommended / defaults to false)
                modules:
                    {
                    # Pre-processing for source
                    source:
                        [
                        {name: load, infer:False, params:{encoding:'utf-8', separator:';'}},
                        {name: missing_values, infer: False, params:{...}},
                        # no recoding will be done as we did not include this module
                        ], 
                    # Pre-processing for ref (only if referential is not internal)
                     (ref:
                         [])
                    # For transformations that apply to both files
                    shared: 
                        {name: dedupe, infer:False, params:{...}}
                    }                
                }
```
