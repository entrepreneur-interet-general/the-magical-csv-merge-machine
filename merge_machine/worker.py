#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:56:10 2017

@author: m75380
"""

import os

import redis
from rq import Worker, Queue, Connection


# Preload necessary imports for efficiency
import api_queued_modules

from admin import Admin
from normalizer import ESNormalizer
from linker import ESLinker


VALID_QUEUES = ['high', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('listen', type=str, nargs='*', default=VALID_QUEUES,
                        help='list of queues to listen to')
    
    args = parser.parse_args()
    listen = args.listen
    
    print(listen)
    
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()