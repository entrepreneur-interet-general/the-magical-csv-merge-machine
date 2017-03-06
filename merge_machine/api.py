#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:01:16 2017

@author: leo


TODO:
    - Safe file name / not unique per date
    
    - API: List of internal referentials
    - API: List of finished modules for given project / source
    - API: List of loaded sources
    
    - API: Fetch infered parameters
    - API: Fetch logs
    - API: Move implicit load out of API
    

DEV GUIDELINES:
    - By default the API will use the file with the same name in the last 
      module that was completed. Otherwise, you can specify the module to use file from
    - Suggestion methods shall be prefixed by suggest (ex: suggest_load_params, suggest_missing_values)
    - Suggestion methods shall can be plugged as input as params variable of transformation modules
    - Single file modules shall take as input: (pandas_dataframe, params)
    - Single file modules suggestion modules shall ouput (params, log)
    - Single file modules replacement modules shall ouput (pandas_dataframe, log)
    
    - Multiple file modules shall take as input: (pd_dataframe_1, pd_dataframe_2, params)
    - Multiple file modules suggestion modules shall ouput params, log
    - Multiple file modules merge module shall ouput ???
    
    - Do NOT return files, instead, user can fetch file through api
    - If bad params are passed to modules, exceptions are raised, it is the APIs role to transform these exceptions in messages
    - Functions to check parameters should be named check_{variable_or_function} (ex: check_file_role)
    - All securing will be done in the API part
    - Always return {"error": ..., "project_id": ..., "response": ...}


NOTES:
    - Pay for persistant storage?

# Transform 
curl -i http://127.0.0.1:5000/transform/ -X POST -F "source=@data/tmp/test_merge.csv" -F "ref=@data/tmp/test_merge_small.csv" -F "request_json=@sample_request.json;type=application/json"

# Upload data
curl -i http://127.0.0.1:5000/download/ -X POST -F "request_json=@sample_download_request.json;type=application/json"

# Download data
curl -i http://127.0.0.1:5000/download/ -X POST -F "request_json=@sample_download_request.json;type=application/json"

# Download metadata
curl -i http://127.0.0.1:5000/metadata/ -X POST -F "request_json=@sample_download_request.json;type=application/json"
"""

import os

from flask import Flask, json, jsonify, redirect, request, url_for, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

from admin import Admin
from project import Project


# Change current path to path of api.py
curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

# Initiate application
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.debug = True
app.secret_key = open('secret_key.txt').read()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Check that files are not too big
          
          
@app.route('/')
@cross_origin()
def index():
    return "Info here: https://github.com/eig-2017/the-magical-csv-merge-machine"
    
def check_request():
    '''Check that input request is valid'''
    pass


def init_project(project_id=None, existing_only=False):
    '''Initialize project and parse request'''  
    
    if (project_id is None) and existing_only:
        raise Exception('Cannot pass None to project_id. No project can be created here')
    proj = Project(project_id)
    
    # Parse json request
    data_params = None
    module_params = None
    if request.json:
        params = request.json
        assert isinstance(params, dict)
    
        if 'data' in params:
            data_params = params['data']
            
            # Make paths secure
            for key, value in data_params.iteritems():
                data_params[key] = secure_filename(value)
            
        if 'params' in params:
            module_params = params['params']
    
    return proj, data_params, module_params
    
@app.route('/project/metadata/<project_id>', methods=['GET', 'POST'])
@cross_origin()
def metadata(project_id):
    '''Fetch metadata for project ID'''
    print project_id
    proj, _, _ = init_project(project_id, True)
    resp = jsonify(error=False,
                   metadata=proj.metadata, 
                   project_id=proj.project_id)
    print resp
    return resp

@app.route('/project/download/<project_id>', methods=['POST'])
@cross_origin()
def download(project_id):
    '''
    Download file from project.
    
    If just project_id: return last modified file
    If variables are specified, return the last file modified with specific variables
    
    request_json = {project_id: ..., file_role: ..., (module: ...), file_name: ...}
    '''
    try:
        proj, data_params, _ = init_project(project_id, True)
        
        if data_params is None:
            data_params = {}
        
        file_role = data_params.get('file_role', None)
        module = data_params.get('module', None)
        file_name = data_params.get('file_name', None)
        
        if file_role is not None:
            file_role = secure_filename(file_role)
        if module is not None:
            module = secure_filename(module)
        if file_name is not None:
            file_name = secure_filename(file_name)
            
        (file_role, module, file_name) = proj.get_last_written(file_role, module, file_name)
            
        if module == 'INIT':
            return jsonify(error=True,
                   message='No changes were made since upload. Download is not \
                           permitted. Please do not use this service for storage')
        
        file_path = proj.path_to(file_role, module, file_name)
        return send_file(file_path)
    
    except Exception, e:
        import pdb
        pdb.set_trace()
        return jsonify(error=True,
                       message=e)

@app.route('/project/upload', methods=['POST'])
@app.route('/project/upload/<project_id>', methods=['POST'])
@cross_origin()
def upload(project_id=None):
    '''
    Uploads source and reference files to project either passed as variable or
    loaded from request parameters
    '''
    if project_id.lower() == 'new_project':
        project_id = None # TODO: Dirty fix bc swagger doesnt take optional path parameters
    
    # Create or Load project
    proj, _ = init_project(project_id)
    
    # 
    for key in ['source', 'ref']:
        if key in request.files:
            file = request.files[key]
            proj.add_init_data(file.stream, key, file.filename)    
    
    return jsonify(error=False,
               metadata=proj.metadata,
               project_id=proj.project_id)



def load_from_params(proj, data_params=None):
    '''Load data to project using the parameters received in request.
    Implicit load is systematic. TODO: Define implicit load
    '''
    if data_params is None:
        file_role = None
        module_name = None
        file_name = None
    else:
        file_role = data_params.setdefault('file_role', None)
        # Skip processing for internal referentials
        if file_role == 'ref' and data_params.setdefault('internal', False):
            raise Exception('Internal data NOT YET IMPLEMENTED')
        
        # Load data from last run (or from user specified)
        file_name = data_params.setdefault('file_name', None)
        module_name = data_params.setdefault('module', None)
        
    if not all(x is not None for x in [file_role, module_name, file_name]):
        (file_role, module_name, file_name) = proj.get_last_written(\
                                        file_role, module_name, file_name)
        
    proj.load_data(file_role, module_name, file_name)



@app.route('/project/run_all/<project_id>', methods=['POST'])
@cross_origin()
def main(project_id):
    '''
    DEPRECATED
    
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
    
    check_request()
    
    #==========================================================================
    # Load project and parameters
    #==========================================================================

    proj, params = init_project(project_id, True)

    #==========================================================================
    # Execute transformations on table(s)
    #==========================================================================
    
    
    for file_role in ['source', 'ref']:

        
        for command in params['modules'][file_role]: 
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





#==============================================================================
# MODULES
#==============================================================================

@app.route('/project/modules/', methods=['GET', 'POST'])
@cross_origin()
def list_modules():
    '''List available modules'''
    return jsonify(error=True,
                   message='This should list the available modules') #TODO: <--


@app.route('/project/modules/infer_mvs/<project_id>', methods=['GET', 'POST'])
@cross_origin()
def infer_mvs(project_id):
    '''Runs the infer_mvs module'''
    proj, data_params, module_params = init_project(project_id, True)
    load_from_params(proj, data_params)
    
    result = proj.infer('infer_mvs', module_params)
        
    # Write log
    proj.write_log_buffer(False)
    
    return jsonify(error=False,
                   response=result)
    
    
@app.route('/project/modules/replace_mvs/<project_id>', methods=['POST'])
@cross_origin()
def replace_mvs(project_id):
    '''Runs the mvs replacement module'''
    proj, data_params, module_params = init_project(project_id, True)
    load_from_params(proj, data_params)
    
    proj.transform('replace_mvs', module_params)

    # Write transformations and log
    proj.write_data()    
    proj.write_log_buffer(True)
    
    return jsonify(error=False)



#==============================================================================
# Admin
#==============================================================================


@app.route('/admin/list_projects', methods=['GET', 'POST'])
@cross_origin()
def list_projects():
    '''Lists all project id_s'''
    admin = Admin()
    list_of_projects = admin.list_projects()
    return jsonify(error=False,
                   response=list_of_projects)




if __name__ == '__main__':
    app.run(debug=True)
