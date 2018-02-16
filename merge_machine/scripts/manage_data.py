#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:37:58 2017

@author: leo
"""

import os
import sys

# Change directory to main folder to be able to access methods
main_dir_path = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(main_dir_path)
os.chdir(main_dir_path)
print(main_dir_path)

try:
    from admin import Admin
except ImportError as e:
    raise ImportError(e.__str__() + '\ntry running from same directory as file')    
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='List or delete projects based on '
                             'time of last use or creation. This script will'
                             ' also delete loose normalization project and indices.',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('request', 
                        type=str,
                        choices=['list', 'delete'],
                        help='Choose what to do with the results')
    
    parser.add_argument('project_type', 
                        type=str,
                        choices=['link', 'normalize'],
                        help='Type of project to delete')
    
    parser.add_argument('-pa', '--project_access', 
                        type=str,
                        choices=['all', 'public', 'private'],
                        default='private',
                        nargs='?',
                        help='Restrictions on project access')
    
    parser.add_argument('-a', '--action', 
                        type=str, 
                        choices=['created', 'last_used'],
                        default='last_used',
                        nargs='?',
                        help='Whether to filter on date of creation or last use')
    
    parser.add_argument('-w', '--when',
                       type=str,
                       choices=['before', 'after'],
                       default='before',
                       nargs='?',
                       help='Choose whether to delete before or after the target' 
                            'timestamp')
    
    parser.add_argument('-hfn', '--hours_from_now', 
                        type=float,
                        default=24*14, # 14 days
                        nargs='?',
                        help='Number of last hours for which to keep data')
    
    parser.add_argument('-ki', '--keep_indices',
                        action='store_true',
                        help='Flag to keep Elasticsearch indices despite having'
                        ' deleted the MMM projects')    
    parser.add_argument('-kll', '--keep_loose_links',
                        action='store_true',
                        help='Flag to NOT delete loose link projects')    

    args = parser.parse_args()
    
    #args = parser.parse_args('list link -pa all -hfn 24 -w before'.split())
    
    # Actual requests
    admin = Admin()
    if args.request == 'list':
        func = admin.list_projects_by_time
    elif args.request == 'delete':
        func = admin.remove_project_by_time
    else:
        raise ValueError('No action associated to request: {0}'.format(args.request))
    
    res = func(args.project_type,
         project_access=args.project_access,
         action=args.action, 
         when=args.when, 
         hours_from_now=args.hours_from_now)
    
    print(res)
    
    if (args.request == 'delete'):
        if not args.keep_indices:
            admin.delete_unused_indices()
        if not args.keep_loose_links:
            admin.delete_loose_links()
