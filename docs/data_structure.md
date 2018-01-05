# Server side file system structure

```
data/
    projects/
        normalize/
            proj_1/ # Unique Id (hash) for each project
                metadata.json # Original file name / list of modules that were completed, in what order / list of file names
                source/
                    INIT/ # The original untransformed files
                        source_name.csv # Initial file uploaded by user
                        infered_config.json # Data from upload (num lines, etc.)
                        source_name.csv__run_info.json # Also data from upload (num lines, etc.)
                    replace_mvs/ # Inference and replacement of missing values
                        source_name.csv # Transformed file
                        infered_config.json # Result of `infer_mvs`
                        source_name.csv__run_info.json # Output of run
                    recode_types/
                        source_name.csv # Transformed file
                        infered_config.json # Result of `infer_types`
                        source_name.csv__run_info.json # Output of run                  
                    concat_with_init/ # Cleaning & normalisation
                        source_name.csv # Initial file uploaded by user
                        source_name.csv__run_info.json 
            proj_2/ # Unique Id (hash) for each project
                [...]


        link/
            proj_A/ # Unique Id (hash) for each project
                metadata.json
                es_linker/
                    column_matches.json
                    labeller.json
                    learned_settings.json
                    source.csv # The merged file
                    source.csv__run_info.json
                results_analysis/
                    infered_config.json # Results of the analysis        
            proj_B/ # Unique Id (hash) for each project
                [...]
     
 ```
