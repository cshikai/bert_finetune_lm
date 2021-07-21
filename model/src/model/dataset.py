from typing import List, Callable
import os
import json
import copy
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from . import transforms
from typing import Dict
import dask.dataframe as dd
import dask

class CovidDataset(Dataset):
    """
    Covid_Dataset Object
    """
    def __init__(self, use_uncased:bool, task:str, mode:str, max_length:int, data_length:int):
        dask.config.set(scheduler='synchronous')
        self.use_uncased = use_uncased
        self.task = task
        self.mode = mode
        self.max_length = max_length
        self.task = task 
        self.data_length = data_length
        mode = self.mode.lower()
        if (self.task == "PRETRAIN"):
            self.path = 'model/'+mode+'_data_transformed_uncased.parquet' if self.use_uncased else 'model/'+mode+'_data_transformed_cased.parquet'
            self.features = ['sentence_a', 'sentence_b', 'labels']
            self.data = dd.read_parquet(self.path, columns=self.features, engine='fastparquet')
        elif (self.task == "QA"):
            self.path = 'model/'+mode+'_qa_data_transformed_uncased.parquet' if self.use_uncased else 'model/'+mode+'_qa_data_transformed_cased.parquet'
            self.features = ['context', 'question', 'answer']
            self.data = dd.read_parquet(self.path, columns=self.features, engine='fastparquet')
        
        self.tokenizer = BertTokenizerFast.from_pretrained('bert_cached/bert-base-uncased') if self.use_uncased else BertTokenizerFast.from_pretrained('bert_cached/bert-base-cased')
        
        
    def __getitem__(self, idx):
        # tokenize data (one sample in the entire dataset (so one seq), not one batch)
        data_transformed = {}
        data_transformed = self.data.loc[idx].compute()
        data_transformed = data_transformed.to_dict('records')
       
        data_tokenized = self.tokenize_steps(data_transformed[0])
        return data_tokenized


    def __len__(self):
        return len(self.data)
    
    
    # Choose tokenizing steps based on task
    def tokenize_steps(self, data_transformed):
        if self.task == "PRETRAIN":
            data_tokenized = self.tokenize_pretrain(data_transformed)
        elif self.task == "QA": 
            data_tokenized = self.tokenize_qa(data_transformed)

        return data_tokenized

    def tokenize_pretrain(self, data_transformed):
        data_tokenized = self.tokenizer(data_transformed['sentence_a'], data_transformed['sentence_b'], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # mlm label
        data_tokenized['labels'] = data_tokenized.input_ids.detach().clone()
        # nsp label
        data_tokenized['next_sentence_label'] = torch.LongTensor([data_transformed['labels']]).T
        ## mask
        # random arr of floats with equal dimensions to input_ids tensor
        rand = torch.rand(data_tokenized.input_ids.shape)
        # mask arr
        # 101 and 102 are the SEP & CLS tokens, don't want to mask them
        mask_arr = (rand * 0.15) * (data_tokenized.input_ids != 101) * (data_tokenized.input_ids != 102) * (data_tokenized.input_ids != 0)
        # assigning masked input ids with 103
        selection = []
        for i in range(data_tokenized.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
            
        for i in range(data_tokenized.input_ids.shape[0]):
            data_tokenized.input_ids[i, selection[i]] = 103
        
        return data_tokenized

     # Tokenize for QA (for one question-answer pair)
    def tokenize_qa(self, data_transformed):
        # edit the context so the answer will always be in the context (since max seq length in bert we're using is 256)
        context = data_transformed['context']
        context_tokens = self.tokenizer(context, return_tensors='pt', truncation=False, padding=False)
        
        answer = data_transformed['answer']
        start_token = context_tokens.char_to_token(answer['answer_start'])
        end_token = context_tokens.char_to_token(answer['answer_end'])
        
        # getting the new context and answer token positions
        # the below are token-wise, so excluding whitespace (note: token-wise != word-wise)
        answer_len = end_token - start_token + 1 # token count
        max_dist = start_token-1 if start_token<=80 else 80
        dist = int(torch.randint(0, max_dist, (1,)))
        answer_start = dist + 1
        answer_end = answer_start + answer_len - 1
        context_start = start_token - dist
        context_end = end_token + int(torch.randint(0, 80, (1,)))
        # if the context start and end token positions exceed the valid range
        if (context_start <= 0):
            context_start = 1
        if (context_end >= len(context_tokens['input_ids'][0])-1):
            context_end = len(context_tokens['input_ids'][0])-2

        new_context = self.tokenizer.decode(context_tokens['input_ids'][0][context_start:context_end+1])
     
        # tokenize using new context
        data_tokenized = self.tokenizer(new_context, data_transformed['question'], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')

        data_tokenized.update({'start_positions': torch.LongTensor([answer_start]), 'end_positions': torch.LongTensor([answer_end])})
        
        # output should have input_ids, token_type_ids, attention_mask, start_positions, end_positions
       
        return data_tokenized

    @staticmethod
    def collate_fn(batch_list):
        # batch_list is a list of dicts, where one dict rep one seq, so need to convert the format into what the model wants
        batch = batch_list[0]
        for i in range(1, len(batch_list)):
            for k, v in batch_list[i].items():
                batch[k] = torch.cat((batch[k], v), 0)
        return batch

