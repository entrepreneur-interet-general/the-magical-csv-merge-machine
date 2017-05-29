#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:41:30 2017

@author: m75380
"""

# TODO: remove this temporary import
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import unittest

import pandas as pd

from infer_nan import infer_mvs

class InferMvsTest(unittest.TestCase) :

    def test_len_ratio(self):
        tab = pd.DataFrame([['City name 1'], ['Another city name'], ['NA'], ['NA'], ['City names are long']], columns=['my_col'])
        
        infered = infer_mvs(tab)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1
        assert infered['columns']['my_col'][0]['val'] == 'NA'

    def test_letter_repetition(self):
        tab_1 = pd.DataFrame([['972'], ['973'], ['91'], ['91'], ['93'], ['000']], columns=['my_col'])
        tab_2 = pd.DataFrame([['972'], ['973'], ['91'], ['91'], ['93'], ['000'], ['111']], columns=['my_col'])
        
        infered = infer_mvs(tab_1)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1
        assert infered['columns']['my_col'][0]['val'] == '000'

        infered = infer_mvs(tab_2)['mvs_dict']
        assert len(infered['columns']) == 0

    def test_punctuation(self):
        tab_1 = pd.DataFrame([['972'], ['973'], ['91'], ['91'], ['-'], ['-']], columns=['my_col'])
        tab_2 = pd.DataFrame([['972'], ['973'], ['91'], ['91'], ['93'], ['-'], ['.']], columns=['my_col'])
        
        infered = infer_mvs(tab_1)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1
        assert infered['columns']['my_col'][0]['val'] == '-'

        infered = infer_mvs(tab_2)['mvs_dict']
        assert len(infered['columns']) == 0


    def test_not_digit(self):
        tab_1 = pd.DataFrame([['4000'], ['0005'], ['6000'], ['91'], ['dontknow']], columns=['my_col'])
        tab_2 = pd.DataFrame([['972'], ['973'], ['91'], ['91'], ['dontknow'], ['anumber'], ['someword']], columns=['my_col'])
        
        infered = infer_mvs(tab_1)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1
        assert infered['columns']['my_col'][0]['val'] == 'dontknow'

        infered = infer_mvs(tab_2)['mvs_dict']
        assert len(infered['columns']) == 0
        
    def test_len_diff(self):
        tab_1 = pd.DataFrame([['0000'], ['0001'], ['0002'], ['0003'], ['1011'], ['00000']], columns=['my_col'])
        tab_2 = pd.DataFrame([['000'], ['001'], ['002'], ['003'], ['111'], ['0'], ['1']], columns=['my_col'])

        infered = infer_mvs(tab_1)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1

        assert infered['columns']['my_col'][0]['val'] == '00000'

        infered = infer_mvs(tab_2)['mvs_dict']
        assert len(infered['columns']) == 0           
        
    def test_usual_forms(self):
        tab_1 = pd.DataFrame([['some'], ['kind'], ['of data'], ['text'], ['has'], ['nan']], columns=['my_col'])
        tab_2 = pd.DataFrame([['none of'], ['them'], ['are'], ['bad'], ['in'], ['this example']], columns=['my_col'])

        infered = infer_mvs(tab_1)['mvs_dict']
        assert len(infered['columns']['my_col']) == 1

        assert infered['columns']['my_col'][0]['val'] == 'nan'

        infered = infer_mvs(tab_2)['mvs_dict']
        print(infered)
        assert len(infered['columns']) == 0           
             

if __name__ == '__main__':
    unittest.main()