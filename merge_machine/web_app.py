#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:20:20 2017

@author: leo

USES: /python-memcached



"""


#SESSION_TYPE = 'MemcachedSessionInterface'

DUMMY_EMIT = {'formated_example': 'EXample 1 \nEXample 2 \n Are the same?',
                     'n_match': 3, 
                     'n_distinct': 5, 
                     'message': 'go sharks', 
                     'has_previous': False}

#@socketio.on('joined', namespace='/')
#def init(deduper=None):
#    print('JOINED !!!')
#    #    emit('message', DUMMY_EMIT)
#    emit('message', flask._app_ctx_stack.labeller.to_emit())




if __name__ == '__main__':
    socketio.run(app)