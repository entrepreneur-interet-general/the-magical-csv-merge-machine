# API calls

## Shared
* new_project: `'/api/new/<project_type>'`
* delete_project: `/api/delete/<project_type>/<project_id>`
* metadata: `/api/metadata/<project_type>/<project_id>`
* get_last_written: `/api/last_written/<project_type>/<project_id>`
* download: `/api/download/<project_type>/<project_id>`
* project_exists: `/api/exists/<project_type>/<project_id>`
* upload_config: `/api/upload_config/<project_type>/<project_id>/`

## Normalize
* upload: `/api/normalize/upload/<project_id>`
* infer_mvs: `/api/normalize/infer_mvs/<project_id>/`
* replace_mvs: `/api/normalize/replace_mvs/<project_id>/`


## Link
* select_file: `/api/link/select_file/<project_id>`
* add_column_matches: `/api/link/add_column_matches/<project_id>/`
* add_column_certain_matches: `/api/link/add_column_certain_matches/<project_id>/`
* add_columns_to_return: `/api/link/add_columns_to_return/<project_id>/<file_role>/`
* linker: `/api/link/dedupe_linker/<project_id>/`









