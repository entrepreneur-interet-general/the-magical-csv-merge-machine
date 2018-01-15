# Scripts

Use this folder to put scripts to manage the API. 

- `api_helpers.py` contains the `APIConnection` class that provides tools to help with API interaction.

- `admin.py` contains methods to list and delete projects, based on their type (link, normalize), creation or last use date as well as tools to delete unused Elasticsearch indices and link projects for which the associated normalization projects no longer exist.

- `upload_referential.py` connect to the API to upload and index files in Elasticsearch, creating public referentials. Configuration files (api connection, and proper way tu upload files) can be found in the `conf` directory. The newly created project ID is referenced by `display_name` in `logs.json`.

- `delete_referential.py` connect to the API to delete a previously uploaded referential (then deleted from `logs.json`).

- `test_api.py` serves as an integration test for the API. It tries to go through all steps of a link project as would a regular user (creating 2 normalization projects, inference, transform, linking...). As of 09/01/18, labelling is not included in the pipeline.
