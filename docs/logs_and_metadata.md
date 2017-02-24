# Logs and metadata formats

## metadata.json

```
{
timestamp: 1487940990.422995 # Creation time
user_id: ??? # If we track users
log:[ # List of modules that were executed with what source files (from what module). was there an error
    {source_file: "source_id_1", module: "load", timestamp: 1487949990.422995, origin: "source_file", error:False},
    {source_file: "source_id_1", module: "missing_values", timestamp: 1487949990.422995, origin: "load", error:False}
    ]
]
}
```

## run_info.json

tbd ?
