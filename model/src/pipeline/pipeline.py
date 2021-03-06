import os
from nltk.probability import DictionaryConditionalProbDist 
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
        self.qa_clean()
       
    def pretrain_clean(self):
        # nltk.download('punkt')
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Download cleaned data from gdrive
        #raw_url = 'https://drive.google.com/uc?id=1yUVHF8Lzvi9gY3YNMjM-n7hlJR7SaG7A' #old url
        # raw_url = 'https://drive.google.com/uc?id=1nq-5XYJ-qEe_WAZDOJuOzQGNbx9P5Wuu'
        output = 'pipeline/data_cleaned.json' 
        # gdown.download(raw_url, output, quiet=False)
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
        outfile.close()

    def qa_clean(self):
        # Download data
        # raw_url = 'https://drive.google.com/uc?id=1SJibr9KlCO89IQiZVMaVum9-iTb27r_s'
        output = 'pipeline/COVID-QA.json'
        # gdown.download(raw_url, output, quiet=False)
        with open(output) as f:
            data = json.load(f)
        
        data_list = []
        data_final = {}

        structured_data = self.structure_qa(data['data'])

        # Split to train test valid
        data_list = self.qa_split_train_valid_test(structured_data)

        # Format data & Append to dict
        data_final['train'] = self.format_qa(data_list['train'])
        data_final['valid'] = self.format_qa(data_list['valid'])
        data_final['test'] = self.format_qa(data_list['test'])

        path = 'pipeline/uncased_qa.json' if self.use_uncased else 'pipeline/cased_qa.json'
        with open(path, 'w') as outfile:
            json.dump(data_final, outfile)
        outfile.close()

    def structure_qa(self, data):
        structured_data = []
        for group in data:
            pair_list = group['paragraphs'][0]['qas']
            context = group['paragraphs'][0]['context']
            for pair in pair_list:
                pair['context'] = context
                structured_data.append(pair)
        return structured_data

    def format_qa(self, data_list):
        data = []

        for pair in data_list:
            question = pair['question']
            context = pair['context']
            for answer in pair['answers']:
                dict_cqa = {}

                # If use uncased, then lowercase everything
                if self.use_uncased:
                    context = context.lower()
                    question = question.lower()
                    answer['text'] = answer['text'].lower()

                # getting start and end indices of the answer in the context
                # if answer begins with a whitespace, increase answer_start by 1
                if (answer['text'][0] == " "):
                    answer['answer_start'] += 1

                answer['text'] = " ".join(answer['text'].split())
                
                gold_text = answer['text']
                start_idx = answer['answer_start']
                gold_text_len = len(gold_text)
                end_idx = start_idx + gold_text_len - 1
                answer['answer_end'] = end_idx

                # remove extra white spaces in the context as well but only after the start_idx
                context = context[0:start_idx] + " ".join(context[start_idx:].split())

                dict_cqa['context'] = context
                dict_cqa['question'] = " ".join(question.split())
                dict_cqa['answer'] = answer
                data.append(dict_cqa)

        return data

    def split_train_valid_test(self, data):
        data_train, data_others = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=False)
        data_valid, data_test = train_test_split(data_others, test_size=0.33, train_size=0.67, shuffle=False)
        split_data = {'train': data_train, 'valid': data_valid, 'test': data_test} # ratio is ~70/20/10
        return split_data

    def qa_split_train_valid_test(self, data):
        data_train, data_others = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True)
        data_valid, data_test = train_test_split(data_others, test_size=0.33, train_size=0.67, shuffle=True)
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
