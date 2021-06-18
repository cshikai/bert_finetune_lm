from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
import os
import numpy as np
import requests
from transformers import BertTokenizer, BertForPreTraining
import itertools
import random

class NSPTokenization():
    def __init__(self, data: list, tokenizer, max_length):
        self.data = data # list of lists of sections where each section is a list of sentences
        self.tokenizer = tokenizer # either cased or uncased tokenizer
        self.max_length = max_length
    def __call__(self):
        sentence_a = []
        sentence_b = []
        labels = []
        for ind, article in enumerate(self.data):
            # list of all sentences in the article
            bag = list(itertools.chain.from_iterable(article))
            bag_size = len(bag)
            # output for each article should be a dict with the keys 'sentence_a', 'sentence_b', 'labels', whose values are the corr lists
            for i, section in enumerate(article): # for each section in one article
                section_size = len(section)
                if (section_size<=1):
                    continue
                else:
                    for j, sent in enumerate(section): # for each sentence in one section
                        if (j <= section_size-2):
                            sentence_a.append(sent)
                            if (random.random() >= 0.5):
                                # sentence_b is the correct next sentence after sentence_a
                                sentence_b.append(section[j+1])
                                labels.append(0)
                            else:
                                # sentence_b is the wrong next sentence after sentence_a
                                rand_sent = sent
                                while (rand_sent == sent):
                                    randi = random.randint(0, bag_size-1)
                                    rand_sent = bag[randi]
                                sentence_b.append(rand_sent)
                                labels.append(1)
                        else: # end of section reached
                            break
                        
        # tokenize
        model_inputs = self.tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # get labels
        model_inputs['labels'] = torch.LongTensor([labels]).T
        
        return model_inputs 

class MLMTokenization():
    def __init__(self, data: list, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self):
        sentence_list = []
        # convert list of articles of sections of sentences to list of sentences
        for article in self.data:
            for section in article:
                for sentence in section:
                    sentence_list.append(sentence)
        
        # tokenize
        model_inputs = self.tokenizer(sentence_list, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # get labels
        model_inputs['labels'] = model_inputs.input_ids.detach().clone()
        ## mask
        # random arr of floats with equal dimensions to input_ids tensor
        rand = torch.rand(model_inputs.input_ids.shape)
        # mask arr
        # 101 and 102 are the SEP & CLS tokens, don't want to mask them
        mask_arr = (rand * 0.15) * (model_inputs.input_ids != 101) * (model_inputs.input_ids != 102) * (model_inputs.input_ids != 0)
        # assigning masked input ids with 103
        selection = []
        for i in range(model_inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
            
        for i in range(model_inputs.input_ids.shape[0]):
            model_inputs.input_ids[i, selection[i]] = 103
       
        return model_inputs

class Tokenization():
    def __init__(self, data, task: str, use_uncased: bool, max_length:int):
        self.data = data
        self.task = task
        self.use_uncased = use_uncased
        self.max_length = max_length
    def __call__(self):
        # define tokenizer
        if self.use_uncased:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # task
        if self.task == "NSP":
            nsp = NSPTokenization(data=self.data, tokenizer=tokenizer, max_length=self.max_length)
            tokenized = nsp()
        elif self.task == "MLM":
            mlm = MLMTokenization(data=self.data, tokenizer=tokenizer, max_length=self.max_length)
            tokenized = mlm()
        else:
            pass # if we decide to fine tune more tasks
            
        return tokenized 

        
class UpperClamp():
    '''
    Clamps the values per column
    '''

    def __init__(self, upper_bound: List[float]) -> None:
        self.upper_bound = torch.Tensor(upper_bound).unsqueeze(0)
    
    def __call__(self,x: Tensor) -> Tensor:
        return torch.min(x, self.upper_bound)


class LowerClamp():
    '''
    Clamps the values per column
    '''
    def __init__(self, lower_bound: List[float]) -> None:
        self.lower_bound = torch.Tensor(lower_bound).unsqueeze(0)
    def __call__(self,x: Tensor) -> Tensor:
        return torch.max(x,self.lower_bound)


class Normalize():
    '''
    normalizes each feature across time. 

    output will be between -1,1
    (sequence,features)
    '''
    def __init__(self, norm_values: List[float]) -> None:
        '''
        norm_values has shape (1,num_features)
        '''
        self.norm_values = np.expand_dims(np.array(norm_values),axis=0)

    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        ''' 
        return (x/self.norm_values)*2 - 1

class NormalizeZeroOne():
    '''
    normalizes each feature across time. 

    output will be between 0,1
    (sequence,features)
    '''
    def __init__(self, norm_values: List[float]) -> None:
        '''
        norm_values has shape (1,num_features)
        '''
        self.norm_values = np.expand_dims(np.array(norm_values),axis=0)

    def __call__(self, x: Tensor) -> Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        ''' 
        return (x/self.norm_values)

class TimeEncoder():

    '''
    This is NOT the exact PE used in transformers, which uses index as position.

    this uses the time in seconds since the start of trajectory as position
    '''
    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
    
    def __call__(self, x):
        '''
        x is a time vector of size [max_len]
        '''
        # make embeddings relatively larger
        max_len = x.shape[0]
        pe = torch.zeros(max_len,self.n_dims)
        for i in range(0, self.n_dims, 2):
            pe[:,i] = torch.sin(x/ (10000 ** ((2 * i)/self.n_dims)))
            pe[:,i+1] = torch.cos(x/ (10000 ** ((2 * (i))/self.n_dims)))           
        return pe

TRANSFORM_MAPPER = {
    'UpperClamp': UpperClamp,
    'LowerClamp': LowerClamp,
    'Normalize': Normalize,
    'NormalizeZeroOne': NormalizeZeroOne,
        }

def get_transforms(transforms_dict):
    x_transforms = []
    for k,params in transforms_dict.items():
        x_transforms.append(TRANSFORM_MAPPER[k](params))
    return x_transforms
