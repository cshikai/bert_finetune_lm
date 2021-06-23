import os 
import pandas as pd
from sqlalchemy import create_engine
import json
from typing import List,Dict
from .config import cfg, Dotdict
import numpy as np
import re
from sklearn.model_selection import train_test_split
import argparse
import nltk
import gdown
import unidecode 


class PMCDataPipeline(object):

    def __init__(self, args):
        self.use_uncased = args.pipeline_use_uncased 
    def __call__(self):
        nltk.download('all')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Download cleaned data from gdrive
        raw_url = 'https://drive.google.com/uc?id=1yUVHF8Lzvi9gY3YNMjM-n7hlJR7SaG7A'
        output = 'data_to_preprocess.json' 
        gdown.download(raw_url, output, quiet=False)
        with open(output) as f:
            data = json.load(f)

        data_list = []

        # Separate the text of each section in 'text' into individual sentences
        for ind, article in enumerate(data):
            for i, text in enumerate(data[ind]['text']):
                data[ind]['text'][i] = re.sub(r'\w[.]\w', '. ', text)
                # upper case everything after ., !, ?
                data[ind]['text'][i] = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), data[ind]['text'][i])
                # split into sentences
                data[ind]['text'][i] = sent_tokenizer.tokenize(data[ind]['text'][i])
        
                if self.use_uncased: # TODO double check if this is correct
                    # lowercase everything and replace accent markers
                    for j, sentence in enumerate(data[ind]['text'][i]):
                        data[ind]['text'][i][j] = unidecode.unidecode(sentence.lower()) 
            
            # remove title, and leave only list of sentences in list of sections in list of articles
            data_list.append(data[ind]['text']) # list structure: [[[]]] articles -> sections -> sentences

        # split into train/test data (split by articles)
        split_data = self.split_train_valid_test(data_list)

        #output to file
        path = 'pipeline/uncased.json' if self.use_uncased else 'pipeline/cased.json'
        with open(path, 'w') as outfile:
            json.dump(split_data, outfile)

    def split_train_valid_test(self, data):
        data_train, data_others = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False)
        data_valid, data_test = train_test_split(data_others, test_size=0.33, train_size=0.67, shuffle=False)
        split_data = {'train': data_train, 'valid': data_valid, 'test': data_test} # ratio is ~70/20/10
        return split_data

    @staticmethod
    def add_pipeline_args(parent_parser):
        def get_unnested_dict(d,root=''):
            unnested_dict = {}
            for key, value in d.items():
                if isinstance(value, Dotdict):
                    unnested_dict.update(get_unnested_dict(value,root+key+'_'))
                else:
                    unnested_dict[root+key]=value
            return unnested_dict
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        unnested_args = get_unnested_dict(cfg,'pipeline_')
        for key,value in unnested_args.items():
            if 'data_maps' not in key:
                parser.add_argument('--'+key,default=value)

        return parser
