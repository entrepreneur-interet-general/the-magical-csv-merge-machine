#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:01:16 2017

@author: leo

Data Structure

data/
    projects/
        proj_1/ # Unique Id for each project (1 referential and 1 type of source)
            source/
                source_id.csv # Initial file uploaded by user
                (source_id_2.csv) # tbd later (if we want to re-use infered and user params)
                metadata.json # Original file name / list of modules that were completed, in what order (last complete filename)
                load/ # Encoding + separator
                    infered_params.json
                    user_params.json
                    source_id.csv # File after transformation
                    run_info.json
                missing_values/
                    infered_params.json
                    user_params.json
                    source_id.csv
                    run_info.json                    
                recoding/ # Cleaning & normalisation
                    infered_params.json
                    user_params.json
                    source_id.csv
                    run_info.json
                (other_pre_processing/)
                    infered_params.json
                    user_params.json
                    source_id.csv
                    run_info.json
        
            (ref/) # Only if user uploads his own referential # Same structure as source/
                ref_id.csv
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
        
    saved_user_data/ # For V2 ? (maybe not file structure ?)
        ???





TODO:
    - Safe file name / not unique per date
    - Generic load module from request + json ()
    - API: List of internal referentials
    - API: List of finished modules for given project / source
    - API: List of loaded sources
    - API: Fetch transformed file
    - API: Fetch infered parameters
    - API: Fetch logs
    

DEV GUIDELINES:
    - Each module shall take care of creating it's own directory
    - ADD: By default the API will use the file with the same name in the last 
            module that was completed. Otherwise, you can specify the module to use file from
    - Method to retrieve log of source / completed modules / date
    - Suggestion methods shall be prefixed by suggest (ex: suggest_load_params, suggest_missing_values)
    - Suggestion methods shall can be plugged as input as params variable of transformation modules
    - Single file modules shall take as input: (pandas_dataframe, params)
    - Single file modules suggestion modules shall ouput params, log
    - Single file modules replacement modules shall ouput pandas_dataframe, log
    
    - Multiple file modules shall take as input: (pd_dataframe_1, pd_dataframe_2, params)
    - Multiple file modules suggestion modules shall ouput params, log
    - Multiple file modules merge module shall ouput ???
    
    - Do NOT return files, instead, user can fetch file through api
    - Should we store intermediate steps?
    - If bad params are passed to modules, exceptions are raised, it is the APIs role to transform these exceptions in messages
    - functions to check parameters should be named check_{variable_or_function} (ex: check_file_role)
    - all securing will be done in the API paért 


NOTES:
    - Pay for persistant storage?



curl -i http://127.0.0.1:5000/new_project/ -X POST -F "source=@data/tmp/test_merge.csv" -F "ref=@data/tmp/test_merge_small.csv" -F "request_json=@sample_request.json;type=application/json"
"""

from flask import Flask, json, jsonify, redirect, request, url_for
from werkzeug.utils import secure_filename

from project import Project

app = Flask(__name__)
app.debug = True
app.secret_key = open('secret_key.txt').read() # TODO: change this, obviously!
#app.config['TMP_UPLOAD'] = '/home/leo/Documents/eig/test_api/data/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Check that files are not too big
          
          
@app.route('/')
def index():
    return "The project is here: https://github.com/eig-2017/the-magical-csv-merge-machine"
    
    
@app.route('/main/', methods=['POST'])
def main():
    '''
    Runs all modules at once (avoids having to call all modules separately + 
    avoids writing unnecessary data).
    
    See https://github.com/eig-2017/the-magical-csv-merge-machine/blob/master/docs/api_calls_schema.md
    for pseudo call
    
    '''
    # Needs: (one file path (source) and one ref_name) or (two file_paths (source + ref))
    # Check that needs are satisfied
    # calls upload_file on one or two files
    # writes paths to source, ref in metadata
    # Use load module (encoding and separator detection) detection and 
    # returns project_id
    
    #==========================================================================
    # Check that form input is valid
    #==========================================================================
    
    # TODO: do this
    
    #==========================================================================
    # Load project 
    #==========================================================================

    params = json.loads(request.files['request_json'].stream.read())
    
    if params['data'].setdefault('project_id', None) is not None:
        proj = Project(params['data']['project_id'])
    else:
        proj = Project()
    
    
    #==========================================================================
    # Load data
    #==========================================================================
    
    for key in ['source', 'ref']:
        if key in request.files:
            file = request.files[key]
            proj.add_init_data(file.stream, key, file.filename)

    #==========================================================================
    # Execute transformations on table(s)
    #==========================================================================
    
    for key in ['source', 'ref']:
        # Skip processing for internal referentials
        if key == 'ref' and params['data'][key].setdefault('internal', False):
            continue
        
        # Load data from last run (or from user specified)
        file_name = params['data'][key]['file_name']
        module_name = params['data'][key].setdefault('module', None)
        if module_name is None:
            module_name = proj.get_last_successful_written_module(key, file_name)
            
        proj.load_data(key, module_name, file_name)
        
        for command in params['modules'][key]: 
            module_name = command['module_name']
            module_params = command['params']
            proj.transform(module_name, module_params)
            
        # Write transformations
        proj.write_data()
    
    #==========================================================================
    # Save files
    #==========================================================================
    
    return jsonify(error=False, 
                   metadata=proj.metadata,
                   project_id=proj.project_id)





if __name__ == '__main__':
    app.run(debug=True)