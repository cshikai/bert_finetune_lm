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
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import unidecode 


class PMCDataPipeline(object):

    def __init__(self):
        self.use_uncased = cfg['use_uncased'] # TODO not so sure about this, need to double check
    def __call__(self):
        # your pipeline code here
        self.use_uncased = cfg['use_uncased'] # TODO not so sure about this, need to double check
        
        nltk.download('all')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Download cleaned data from gdrive
        raw_url = 'https://drive.google.com/uc?id=1yUVHF8Lzvi9gY3YNMjM-n7hlJR7SaG7A'
        output = 'data_to_preprocess.json' # TODO check where to download data to
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
        split_data = self.split_train_test(data_list)

        if self.use_uncased:
            # output to a file
            with open('uncased.json', 'w') as outfile:
                json.dump(split_data, outfile)
            # # login
            # self.login()
            # # upload file
            # self.uploadfile("uncased.json")
        else:
            # output to a file
            with open('cased.json', 'w') as outfile:
                json.dump(split_data, outfile)
            # # login
            # self.login()
            # # upload file
            # self.uploadfile("cased.json")

    def split_train_test(self, data):
        data_train, data_test = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False)
        split_data = {'train': data_train, 'test': data_test}
        return split_data

    ## For uploading to grive
    # def login():
    #     global gauth, drive
    #     gauth = GoogleAuth()
    #     # Creates local webserver and auto handles authentication
    #     gauth.LocalWebserverAuth() 
    #     drive = GoogleDrive(gauth) 

    # def uploadfile(filename):
    #     # Get parent folder    
    #     gfile = drive.CreateFile({'parents': [{'id': '1dxQeB6hVfvSIFUy64L1bnX668ehgX5nC'}]})
    #     # Read file and set it as the content of this instance
    #     gfile.SetContentFile(filename)
    #     # Upload the file
    #     gfile.Upload() 

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
