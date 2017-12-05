# API calls

## API

* ping flask API: `/api/ping`
* ping redis queue: `/api/ping_redis`

## Generic API methods (normalize and link)

* create a new project: `/api/new/<project_type>`
* delete a project: `/api/delete/<project_type>/<project_id>`
* fetch project metadata: `/api/metadata/<project_type>/<project_id>`
* skip a logged step: `/api/set_skip/<project_type>/<project_id>`
* get the identifiers for the last data written: `/api/last_written/<project_type>/<project_id>`
* download a data file: `/api/download/<project_type>/<project_id>`
* get a sample of data: `/api/sample/<project_type>/<project_id>`
* check if a project exists: `/api/exists/<project_type>/<project_id>`
* generic config read: `/api/download_config/<project_type>/<project_id>/`
* generic config upload: `/api/upload_config/<project_type>/<project_id>/`

## Normalize API methods

* select columns to clean: `/api/normalize/select_columns/<project_id>`
* upload a new data file: `/api/normalize/upload/<project_id>`
* make a mini: `/api/normalize/make_mini/<project_id>`

## Link API Methods

* select a normalization project to use as source or reference: `/api/link/select_file/<project_id>`
* add column matches to use for linking: `/api/link/add_column_matches/<project_id>/`
* add certain column matches to use for auto-learn: `/api/link/add_column_certain_matches/<project_id>/`
* add a labelled pair: `/api/link/label_pair/<project_id>/`

## Socket methods
 
* load a labeller object: `load_labeller`
* update the labeller based on user input: `answer`
* update the labeller filters: `update_filters`
* write learned settings upon training completion: `complete_training`
* terminate ?: `terminate`

## ES fetch

* fetch document in ES by ID  `/api/es_fetch_by_id/<project_type>/<project_id>`

## Scheduler

### List of schdulable jobs 
```
  'infer_mvs': {'project_type': 'normalize'}, 
  'replace_mvs': {'project_type': 'normalize'}, 
  'infer_types': {'project_type': 'normalize'}, 
  'recode_types': {'project_type': 'normalize'}, 
  'concat_with_init': {'project_type': 'normalize'}, 
  'run_all_transforms': {'project_type': 'normalize'}, 
  'create_es_index': {'project_type': 'link'},
  'create_es_labeller': {'project_type': 'link', 
                      'priority': 'high'}, 
  'es_linker': {'project_type': 'link'},
  'infer_restriction': {'project_type': 'link', 
                        'priority': 'high'}, 
  'perform_restriction': {'project_type': 'link'},
  'linker': {'project_type': 'link'}, 
  'link_results_analyzer': {'project_type': 'link'}
```
* schedule a job: `/api/schedule/<job_name>/<project_id>/`
* get job result: `/queue/result/<job_id>`
* cancel job in redis queue: `/queue/cancel/<job_id>`
* count jobs in redis queue: `/queue/num_jobs/`
* count jobs in redis queue befor given id: `/queue/num_jobs/<job_id>`

## Admin

* list public projects ids: `/api/public_project_ids/<project_type>`
* list public projects metadata: `/api/public_projects/<project_type>`







