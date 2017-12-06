#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:10:02 2017

@author: m75380

Tools to help interact with the API

"""
import json
import pprint
import requests
import time


class APIConnection():
    '''Connection to an API and display of calls''' 
    
    def __init__(self, protocol, host, print_=True, print_char_limit = 10000):
        self.protocol = protocol
        self.host = host
        self.print_ = print_
        self.print_char_limit = print_char_limit
        
        
    def my_pformat(self, dict_obj):
        formated_string = pprint.pformat(dict_obj)
        if len(formated_string) > self.print_char_limit:
            formated_string = formated_string[:self.print_char_limit]
            formated_string += '\n[ ... ] (increase PRINT_CHAR_LIMIT to see more...)'
        return formated_string
    
    def my_print(func):
        def wrapper(self, *args, **kwargs):
            if self.print_:
                print('\n' + '>'*60 + '\n', args[0])
                
                if len(args) >= 2:
                    body = args[1]
                    if body:
                        print('\n <> POST REQUEST:\n', self.my_pformat(body))
            resp = func(self, *args, **kwargs)
            
            if self.print_:
                print('\n <> RESPONSE:\n', self.my_pformat(resp))        
            
            return resp
        return wrapper            
        
    @my_print
    def get_resp(self, url_to_append):
        url = self.protocol + self.host + url_to_append
        resp = requests.get(url)
        
        if resp.ok:
            parsed_resp = json.loads(resp.content.decode())
            #        _print(url_to_append,  parsed_resp)
            return parsed_resp
        else: 
            raise Exception('Problem:\n', resp)
    
    @my_print
    def post_resp(self, url_to_append, body, **kwargs):
        url = self.protocol + self.host + url_to_append
        resp = requests.post(url, json=body, **kwargs)     
        
        if resp.ok:
            parsed_resp = json.loads(resp.content.decode())
            #        _print(url_to_append, parsed_resp)
            return parsed_resp
        else: 
            raise Exception('Problem:\n', resp, '\nContent:', resp.content, '\nUrl:', url)
    
    @my_print
    def post_download(self, url_to_append, body, **kwargs):
        url = self.protocol + self.host + url_to_append
        resp = requests.post(url, json=body, **kwargs)     
        
        if resp.ok:
            return resp
        else: 
            raise Exception('Problem:\n', resp)    
    
    def wait_get_resp(self, url_to_append, max_wait=30):
        url = self.protocol + self.host + url_to_append
        print('this_url', url)
        start_time = time.time()
        while (time.time() - start_time) <= max_wait:
            resp = requests.get(url)
            if resp.ok:
                parsed_resp = json.loads(resp.content.decode())
            else: 
                raise Exception('Problem:\n', resp)
                
            if parsed_resp['completed']:
                if self.print_:
                    print('\n <> RESPONSE AFTER JOB COMPLETION (Waited {0} seconds):'.format(time.time()-start_time))
                    print(self.my_pformat(parsed_resp))
                return parsed_resp
            time.sleep(0.25)
        print(time.time() - start_time)
        raise Exception('Timed out after {0} seconds'.format(max_wait))