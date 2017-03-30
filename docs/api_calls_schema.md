# Pseudo API request for the main recoding service [OBSOLETE]

Use `POST` request to post the following fields:

- `request_json`: (see below) json with the run parameters

```
request_json = {
                # What data to use / can be combined with passing file through POST
                data:{
                      project_id:ABC, 
                      # Use source from given project, as was computed in module (or by default, in last module)
                      source: {file_name: my_source, (module:example_module)}, 
                      ref: {file_name: sirene, internal:True}
                      }
                
                # Module params (indicates what modules to use, with what parameters) 
                # (infer:True will use our own suggestions / not recommended / defaults to false)
                modules:
                    {
                    # Pre-processing for source
                    source:
                        [
                        {module_name: load, infer:False, params:{encoding:'utf-8', separator:';'}},
                        {module_name: missing_values, infer: False, params:{...}},
                        # no "recoding" will be done as we did not include this module
                        ], 
                    # Pre-processing for ref (only if referential is not internal)
                     (ref:
                         [])
                    # For transformations that apply to both files
                    shared: 
                        {module_name: dedupe, infer:False, params:{...}}
                    }                
                }
```

# Upload

- `source`: (optional) csv file to upload to source
- `ref`: (optional) csv file to upload to reference
