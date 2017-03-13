#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:20:20 2017

@author: leo
"""

import os

from flask import Flask, render_template, session
from flask_socketio import emit, SocketIO


# Change current path to path of web_app.py
curdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curdir)

app = Flask(__name__)
app.config['SECRET_KEY'] = open('secret_key.txt').read()
socketio = SocketIO(app)


def init_var():
    a = 1
    return a


@socketio.on('joined', namespace='/')
def init(deduper=None):
    print('JOINED !!!')
    
    #session['labeller'] = Labeller(deduper)
    #session['labeller'].new_label()
    
    emit('message', {'test_var': 'yolozer'})
    #emit('message', session['labeller'].to_emit())

@socketio.on('answer', namespace='/')
def get_answer(user_input):
    print('I am here')
    message = ''
    if session['labeller'].answer_is_valid(user_input):
        session['labeller'].parse_valid_input(user_input)
        if session['labeller'].finished:
            # TODO: deal with this (write train and redirect)
            message = 'Sent an invalid answer'
        else:
            session['labeller'].new_label()
    
    emit('message', session['labeller'].to_emit(message=message))
    
    

@app.route('/', methods=['GET'])
def main():    
    return render_template('dedupe_training.html', test_var='YOLO') 
    return render_template('dedupe_training.html',
                       formated_example='EXample 1 \nEXample 2 \n Are the same?',
                       n_match=4,
                       n_distinct=6,
                       message='',
                       has_previous=False)



    return render_template('dedupe_training.html',
                       formated_example=session['labeller'].formated_example,
                       n_match=session['labeller'].n_match,
                       n_distinct=session['labeller'].n_distinct,
                       message='',
                       has_previous=False)




#==============================================================================
# Deduper
#==============================================================================


def unique(seq) :
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

class Labeller():
    def __init__(self, deduper):
        self.finished = False
        self.use_previous = False
        self. fields = unique(field.field 
                              for field
                              in deduper.data_model.primary_fields)
        self.buffer_len = 1
        self.examples_buffer = []
        self.uncertain_pairs = []

    def answer_is_valid(self, user_input):
        '''Check if the user input is valid'''
        if self.examples_buffer:
            valid_responses = {'y', 'n', 'u', 'f', 'p'}
        else: 
            valid_responses = {'y', 'n', 'u', 'f'}
        
        return user_input in valid_responses
        
    def to_emit(self, message=''):
        '''Creates a dict to be sent to the template'''
        dict_to_emit = dict()
        dict_to_emit['formated_example'] = self._format_fields()
        dict_to_emit['n_match'] = session['labeller'].n_match,
        dict_to_emit['n_distinct'] = session['labeller'].n_distinct,
        dict_to_emit['has_previous'] = False
        dict_to_emit['message'] = message


    def _format_fields(self):
        '''Return string containing fields and field names'''
        # TODO: This should be done in template
        self.formated_example = ''
        for pair in self.record_pair:
            for field in self.fields:
                line = "%s : %s" % (field, pair[field])
                self.formated_example += line + '\n'
            
        
    def new_label(self):
        if self.use_previous:
            record_pair, _ = self.examples_buffer.pop(0)
            self.use_previous = False
        else:
            if not self.uncertain_pairs:
                uncertain_pairs = self.deduper.uncertainPairs()
            self.record_pair = uncertain_pairs.pop()
                     
        self.n_match = (len(self.deduper.training_pairs['match']) +
                   sum(label=='match' for _, label in self.examples_buffer))
        self.n_distinct = (len(self.deduper.training_pairs['distinct']) +
                      sum(label=='distinct' for _, label in self.examples_buffer))

        self.user_input = ''        

        

    def parse_valid_answer(self, user_input):
        if user_input == 'y':
            self.examples_buffer.insert(0, (self.record_pair, 'match'))
        elif user_input == 'n' :
            self.examples_buffer.insert(0, (self.record_pair, 'distinct'))
        elif user_input == 'u':
            self.examples_buffer.insert(0, (self.record_pair, 'uncertain'))
        elif user_input == 'f':
            self.finished = True
        elif user_input == 'p':
            self.use_previous = True
            self.uncertain_pairs.append(self.record_pair)
        
        if len(self.examples_buffer) > self.buffer_len:
            record_pair, label = self.examples_buffer.pop()
            if label in ['distinct', 'match']:
                examples = {'distinct' : [], 'match' : []}
                examples[label].append(record_pair)
                self.deduper.markPairs(examples)


#def consoleLabel(deduper): # pragma: no cover
#    '''
#    Command line interface for presenting and labeling training pairs
#    by the user
#    
#    Argument :
#    A deduper object
#    '''

#    finished = False
#    use_previous = False
#    fields = unique(field.field
#                    for field
#                    in deduper.data_model.primary_fields)
#
#    buffer_len = 1 # Max number of previous operations
#    examples_buffer = []
#    uncertain_pairs = []
    
#    if use_previous:
#        record_pair, _ = examples_buffer.pop(0)
#        use_previous = False
#    else:
#        if not uncertain_pairs:
#            uncertain_pairs = deduper.uncertainPairs()
#        record_pair = uncertain_pairs.pop()
#                 
#    n_match = (len(deduper.training_pairs['match']) +
#               sum(label=='match' for _, label in examples_buffer))
#    n_distinct = (len(deduper.training_pairs['distinct']) +
#                  sum(label=='distinct' for _, label in examples_buffer))
#    
#    for pair in record_pair:
#        for field in fields:
#            line = "%s : %s" % (field, pair[field])
#            print(line, file=sys.stderr)
#        print(file=sys.stderr) 

#    print("{0}/10 positive, {1}/10 negative".format(n_match, n_distinct),
#            file=sys.stderr)
#    print('Do these records refer to the same thing?', file=sys.stderr)
    
#    valid_response = False
#    user_input = ''    

#    while not valid_response:
#        if examples_buffer:
#            prompt = '(y)es / (n)o / (u)nsure / (f)inished / (p)revious'
#            valid_responses = {'y', 'n', 'u', 'f', 'p'}
#        else: 
#            prompt = '(y)es / (n)o / (u)nsure / (f)inished'
#            valid_responses = {'y', 'n', 'u', 'f'}
#
#        print(prompt, file=sys.stderr)
#        user_input = input()
#        if user_input in valid_responses:
#            valid_response = True


#    if user_input == 'y':
#        examples_buffer.insert(0, (record_pair, 'match'))
#    elif user_input == 'n' :
#        examples_buffer.insert(0, (record_pair, 'distinct'))
#    elif user_input == 'u':
#        examples_buffer.insert(0, (record_pair, 'uncertain'))
#    elif user_input == 'f':
#        print('Finished labeling', file=sys.stderr)
#        finished = True
#    elif user_input == 'p':
#        use_previous = True
#        uncertain_pairs.append(record_pair)
#    
#    if len(examples_buffer) > buffer_len:
#        record_pair, label = examples_buffer.pop()
#        if label in ['distinct', 'match']:
#            examples = {'distinct' : [], 'match' : []}
#            examples[label].append(record_pair)
#            deduper.markPairs(examples)












if __name__ == '__main__':
    socketio.run(app)