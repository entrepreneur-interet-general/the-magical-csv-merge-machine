# Server side file system structure [DEPRECATED]
### TODO: Change to unique file transform + linking project

```
data/
    projects/
        proj_1/ # Unique Id for each project (1 referential and 1 type of source)
            metadata.json # Original file name / list of modules that were completed, in what order / list of file names
            source/
                INIT/ # Raw files as they were uploaded
                    source_name.csv # Initial file uploaded by user
                    (source_2.csv) # Tbd later (if we want to re-use infered and user params)
                    run_info.json
                load/ # Encoding + separator
                    infered_params.json
                    user_params.json
                    source_name.csv # File after transformation
                    run_info.json
                missing_values/
                    infered_params.json
                    user_params.json
                    source_name.csv
                    run_info.json                    
                recoding/ # Cleaning & normalisation
                    infered_params.json
                    source_name.json
                    source.csv
                    run_info.json
                (other_pre_processing/)
                    infered_params.json
                    user_params.json
                    source_name.csv
                    run_info.json
        
            (ref/) # Only if user uploads his own referential # Same structure as source/
                INIT/
                    ref.csv
                [...] # Same as source
                
            dedupe/
                infered_params.json # What columns to pair
                user_params.json # What columns to pair
                user_training.json # Training pairs
                learned_params.txt 
                other_dedupe_generated_files.example 
                merged_id.csv
                
            analysis/
                analysis.json
    
        proj_2/
            [...]
        [...]
        
    referentials/ # Internal referentials
        ref_name_1/ # For example "sirene_restreint" # Same structure as source/
            ref.csv
            [...] # Same as source
        ref_name_2/
            [...]
        [...]
    users/ # To be defined ?
        ???
    saved_user_data/ # For V2 ?
        ???
 ```
