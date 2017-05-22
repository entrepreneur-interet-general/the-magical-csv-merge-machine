#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:56:10 2017

@author: m75380
"""

import os

import redis
from rq import Worker, Queue, Connection

listen = ['default']
    
redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()