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


class PMCDataPipeline(object):

    def __init__(self):
        self.use_uncased = cfg['use_uncased'] # TODO not so sure about this, need to double check
    def __call__(self):
        # your pipeline code here
        nltk.download(all)

        # Download cleaned data from gdrive
        raw_url = 'https://drive.google.com/uc?id=1yUVHF8Lzvi9gY3YNMjM-n7hlJR7SaG7A'
        output = '/home/dh/Desktop/data_to_preprocess' # TODO check where to download data to
        gdown.download(raw_url, output, quiet=False)
        with open(output) as f:
            data = json.load(f)

        # Separate the text of each section in 'text' into individual sentences
        for ind, article in enumerate(data):
            for i, text in enumerate(data[ind]['text']):
                data[ind]['text'][i] = tokenizer.tokenize(text)
        
        if self.use_uncased: # TODO double check if this is correct
            # lowercase everything and replace accent markers
            # upload this data to gdrive
            pass
        else:
            # upload unchanged data (with cases and accent markers retained) to gdrive
            pass
    

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
