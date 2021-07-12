from typing import List, Callable
import os
import json
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
# from datasets import Dataset
from transformers import BertTokenizerFast
#import dask.dataframe as dd
from . import transforms
from typing import Dict
import ast

class CovidDataset(Dataset):
    """
    Covid_Dataset Object
    """
    def __init__(self, use_uncased:bool, task:str, mode:str, max_length:int, data_length:int):
        self.use_uncased = use_uncased
        self.task = task
        self.mode = mode
        self.max_length = max_length
        self.task = task 
        self.data_length = data_length

        # if self.task == "QA":
        #     # path for QA
        #     path = 'pipeline/uncased_qna.json' if self.use_uncased else 'pipeline/cased_qna.json'
        # elif self.task == "PRETRAIN":
        #     # paths for nsp and mlm
        #     path = 'pipeline/uncased.json' if self.use_uncased else 'pipeline/cased.json'

        # with open(path) as f:
        #     all_data = json.load(f)
        #     self.data_loaded = all_data[self.mode]


        # transformation = transforms.Transformations(data=self.data_loaded, task=self.task)

        # self.data_transformed = transformation()

        if (self.task == "QA"):
            for i,ans in enumerate(self.data_transformed['answers']):
                if (('answer_end' not in ans.keys()) or ('answer_start' not in ans.keys())):
                    self.data_transformed['contexts'].pop(i)
                    self.data_transformed['questions'].pop(i)
                    self.data_transformed['answers'].pop(i)

        self.tokenizer = BertTokenizerFast.from_pretrained('bert_cached/bert-base-uncased') if self.use_uncased else BertTokenizerFast.from_pretrained('bert_cached/bert-base-cased')
        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') if self.use_uncased else BertTokenizerFast.from_pretrained('bert-base-cased')

        
        
    def __getitem__(self, idx):
        # tokenize data (one sample in the entire dataset (so one seq), not one batch)
        data_transformed = {}
        mode = self.mode.lower()
        path = 'model/'+mode+'_data_transformed_uncased.txt' if self.use_uncased else 'model/'+mode+'_data_transformed_cased.txt'

        with open(path) as fp:
            for i, line in enumerate(fp):
                if i == idx:
                    data_transformed = ast.literal_eval(line)
                    break
        data_tokenized = self.tokenize_steps(data_transformed)
        return data_tokenized


    def __len__(self):
        
        return self.data_length
        # if self.task == "PRETRAIN":
        #     return len(self.data_transformed['sentence_a'])
        # elif self.task == "QA":
        #     return len(self.data_transformed['questions'])
    
    
    # Choose tokenizing steps based on task
    def tokenize_steps(self, data_transformed):
        if self.task == "PRETRAIN":
            data_tokenized = self.tokenize_pretrain(data_transformed)
        elif self.task == "QA": 
            data_tokenized = self.tokenize_qna(data_transformed)

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

    # Tokenize for NSP
    # def tokenize_nsp(self, idx):
    #     # tokenize
    #     data_tokenized = self.tokenizer(self.data_transformed['sentence_a'][idx], self.data_transformed['sentence_b'][idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
    #     # post tokenize
    #     data_tokenized['labels'] = torch.LongTensor([self.data_transformed['labels'][idx]]).T
    #     return data_tokenized

    #  # Tokenize for MLM
    # def tokenize_mlm(self, idx):
    #     # tokenize
    #     data_tokenized = self.tokenizer(self.data_transformed['sentence_list'][idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
    #     # post tokenize
    #     # get labels
    #     data_tokenized['labels'] = data_tokenized.input_ids.detach().clone()
    #     ## mask
    #     # random arr of floats with equal dimensions to input_ids tensor
    #     rand = torch.rand(data_tokenized.input_ids.shape)
    #     # mask arr
    #     # 101 and 102 are the SEP & CLS tokens, don't want to mask them
    #     mask_arr = (rand * 0.15) * (data_tokenized.input_ids != 101) * (data_tokenized.input_ids != 102) * (data_tokenized.input_ids != 0)
    #     # assigning masked input ids with 103
    #     selection = []
    #     for i in range(data_tokenized.input_ids.shape[0]):
    #         selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
            
    #     for i in range(data_tokenized.input_ids.shape[0]):
    #         data_tokenized.input_ids[i, selection[i]] = 103
        
    #     return data_tokenized
        

     # Tokenize for QNA (for one question-answer pair)
    def tokenize_qna(self, idx):
        # edit the context so the answer will always be in the context
        context = self.data_transformed['contexts'][idx]
        context_tokens = self.tokenizer(context, return_tensors='pt', truncation=False, padding=False)
        
        answer = self.data_transformed['answers'][idx]
        start_token = context_tokens.char_to_token(answer['answer_start'])
        end_token = context_tokens.char_to_token(answer['answer_end'])
        
        # getting the new context and answer token positions
        # the below are token-wise, so excluding whitespace (note: token-wise != word-wise)
        answer_len = end_token - start_token + 1 # token count
        max_dist = start_token-1 if start_token<=80 else 80
        dist = torch.randint(0, max_dist)
        answer_start = dist + 1
        answer_end = answer_start + answer_len - 1
        context_start = start_token - dist
        context_end = end_token + torch.randint(0, 80)
        # if the context start and end token positions exceed the valid range
        if (context_start <= 0):
            context_start = 1
        if (context_end >= len(context_tokens['input_ids'][0])-1):
            context_end = len(context_tokens['input_ids'][0])-2

        new_context = self.tokenizer.decode(context_tokens['input_ids'][0][context_start:context_end+1])
     
        # tokenize using new context
        data_tokenized = self.tokenizer(new_context, self.data_transformed['questions'][idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        
        # adding start and end position of the answer to data_tokenized
        # start_position = data_tokenized.char_to_token(answer_start)
        # end_position = data_tokenized.char_to_token(answer_end)

        # if start/end position is None, the answer passage has been truncated
        # if (start_position is None):
        #     start_position = self.max_length
        # if (end_position is None):
        #     end_position = self.max_length

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

