# Logs and metadata formats

## metadata.json

```
{
timestamp: 1487940990.422995, # Creation time
user_id: ???, # If we track users
use_internal_ref: True,
internal_ref_name: sirene,
source_names = ['source_1.csv']
source_log:[ # List of modules that were executed with what source files (from what module). was there an error
    {file_name: "source_id_1", module: "load", timestamp: 1487949990.422995, origin: "INIT", error:False, error_msg=None, written: False},
    {file_name: "source_id_1", module: "missing_values", timestamp: 1487949990.422995, origin: "load", error:False, error_msg=None, written: True}
    ]
ref_log:[]
metadata['project_id'] = 347a7ba113a8cb8023b0c40246ec9098
]
}
```

NB: `origin` refers to the name of the module previously run. If `origin:INIT`, this means the module was run using the file uploaded by the user. 
NB2: `written` means the file was actually written after the step was completed (rather than just kept in memory)

## run_info.json

tbd ?
