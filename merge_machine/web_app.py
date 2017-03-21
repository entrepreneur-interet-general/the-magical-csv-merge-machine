#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:20:20 2017

@author: leo

USES: /python-memcached

TODO: Gray out next button until all fields are properly filled
TODO: Do not go to next page if an error occured

"""

import gc
import os

import flask
from flask import Flask, render_template, session, url_for
from flask_session import Session, MemcachedSessionInterface
from flask_socketio import emit, SocketIO
import pandas as pd

from dedupe_linker import format_for_dedupe, load_deduper
from labeller import Labeller

# Change current path to path of web_app.py
curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

app = Flask(__name__)

app.config['SESSION_TYPE'] = "memcached"# 'memcached'

Session(app)
app.config['SECRET_KEY'] = open('secret_key.txt').read()

socketio = SocketIO(app)

#SESSION_TYPE = 'MemcachedSessionInterface'

DUMMY_EMIT = {'formated_example': 'EXample 1 \nEXample 2 \n Are the same?',
                     'n_match': 3, 
                     'n_distinct': 5, 
                     'message': 'go sharks', 
                     'has_previous': False}


GLOBAL_TEST = {'rappers':['jadakis', '50 cent']}
#@socketio.on('joined', namespace='/')
#def init(deduper=None):
#    print('JOINED !!!')
#    #    emit('message', DUMMY_EMIT)
#    emit('message', flask._app_ctx_stack.labeller.to_emit())

@socketio.on('answer', namespace='/')
def get_answer(user_input):
    # TODO: avoiid multiple click
    message = ''
    #message = 'Expect to have about 50% of good proposals in this phase. The more you label, the better...'
    if flask._app_ctx_stack.labeller.answer_is_valid(user_input):
        flask._app_ctx_stack.labeller.parse_valid_answer(user_input)
        if flask._app_ctx_stack.labeller.finished:
            print('Writing train')
            flask._app_ctx_stack.labeller.write_training(flask._app_ctx_stack.training_path)
            print('Wrote train')

            emit('redirect', {'url': url_for('done')})
        else:
            flask._app_ctx_stack.labeller.new_label()
    else:
        message = 'Sent an invalid answer'
    emit('message', flask._app_ctx_stack.labeller.to_emit(message=message))
    

@app.route('/download_page', methods=['GET'])
def done():
    return 'DONE'


@app.route('/', methods=['GET'])
def main():
    
    import json
    with open('local_test_data/rnsr/my_dedupe_rnsr_config.json') as f:
       my_config = json.load(f)    
    paths = my_config['paths']
    params = my_config['params']    
    
    my_variable_definition = params['variable_definition']      
    flask._app_ctx_stack.training_path = paths['train']

    ref_path = paths['ref']
    source_path = paths['source']      
    
    # Put to dedupe input format
    ref = pd.read_csv(ref_path, encoding='utf-8', dtype='unicode')
    data_ref = format_for_dedupe(ref, my_variable_definition, 'ref') 
    del ref # To save memory
    gc.collect()
    
    # Put to dedupe input format
    source = pd.read_csv(source_path, encoding='utf-8', dtype='unicode')
    data_source = format_for_dedupe(source, my_variable_definition, 'source')
    del source
    gc.collect()
    
    #==========================================================================
    # Should really start here
    #==========================================================================
    deduper = load_deduper(data_ref, data_source, my_variable_definition)

    flask._app_ctx_stack.labeller = Labeller(deduper)
    flask._app_ctx_stack.labeller.new_label()
    
    print(flask._app_ctx_stack.labeller.to_emit(''))
    return render_template('dedupe_training.html', **flask._app_ctx_stack.labeller.to_emit(''))
    #return render_template('dedupe_training.html', **DUMMY_EMIT)



if __name__ == '__main__':
    socketio.run(app)