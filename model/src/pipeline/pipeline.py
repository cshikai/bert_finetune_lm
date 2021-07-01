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
        self.pretrain_clean()
        self.qna_clean()
       
    def pretrain_clean(self):
        nltk.download('all')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Download cleaned data from gdrive
        #raw_url = 'https://drive.google.com/uc?id=1yUVHF8Lzvi9gY3YNMjM-n7hlJR7SaG7A' #old url
        raw_url = 'https://drive.google.com/uc?id=1nq-5XYJ-qEe_WAZDOJuOzQGNbx9P5Wuu'
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
        
                if self.use_uncased: 
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

    def qna_clean(self):
        # Download data
        raw_url = 'https://drive.google.com/uc?id=1SJibr9KlCO89IQiZVMaVum9-iTb27r_s'
        output = 'data_to_preprocess_qa.json'
        gdown.download(raw_url, output, quiet=False)
        with open(output) as f:
            data = json.load(f)
        
        data_list = []
        data_final = {}

        # Split to train test valid
        data_list = self.split_train_valid_test(data['data'])

        # Format data & Append to dict
        data_final['train'] = self.format_qna(data_list['train'])
        data_final['valid'] = self.format_qna(data_list['valid'])
        data_final['test'] = self.format_qna(data_list['test'])

        # TODO: Check if qna needs cased and uncased
        path = 'pipeline/uncased_qna.json' if self.use_uncased else 'pipeline/cased_qna.json'
        with open(path, 'w') as outfile:
            json.dump(data_final, outfile)

    def format_qna(self, data_list):
        data = []

        # Append to data list
        for group in data_list:  
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        dict_cqa = {}

                        # If uncased, then lowercase everything. 
                        if self.use_uncased:
                            context = context.lower()
                            question = question.lower()
                            answer = answer.lower()
                         
                        dict_cqa['context'] = context
                        dict_cqa['question'] = question
                        dict_cqa['answer'] = answer
                        data.append(dict_cqa)

        # getting start and end indices of the answer in the context
        for ind, pair in enumerate(data):
            # Get rid of trailing/extra spaces
            data[ind]['answer']['text'] = " ".join(data[ind]['answer']['text'].split())
            answer = data[ind]['answer']
            context = data[ind]['context']
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            if context[start_idx:end_idx] == gold_text:
                data[ind]['answer']['answer_end'] = end_idx
            # in case index is off by 1 or 2
            # When the gold label is off by one character
            elif context[start_idx-1:end_idx-1] == gold_text:
                data[ind]['answer']['answer_start'] = start_idx - 1
                data[ind]['answer']['answer_end'] = end_idx - 1   
            # When the gold label is off by two characters
            elif context[start_idx-2:end_idx-2] == gold_text:
                data[ind]['answer']['answer_start'] = start_idx - 2
                data[ind]['answer']['answer_end'] = end_idx - 2     

        return data

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
