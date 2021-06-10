from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
import os
import numpy as np
import gdown
import requests
from transformers import BertTokenizer, BertForPreTraining
import itertools
import random


class NSPLabels(object):
    def __init__(self, data):
        self.data = data # list of lists of sections where each section is a list of sentences
    def __call__(self, data):
        self.data = data
        
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

        # store data (paired sentences) for all articles as a dictionary
        self.data = {'sentence_a': sentence_a, 'sentence_b': sentence_b, 'labels': labels}
        
        return self.data 

# class Tokenization(object):
#     def __init__(self, data, task, use_uncase):
#         self.data = data
#         self.task = task
#         self.use_uncase = use_uncase
#     def __call__(self, data, task, use_uncase):
#         self.data = data
#         self.task = task
#         self.use_uncase = use_uncase

#         if use_uncase:
#             tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         else:
#             tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#         if task == ""
#         sentence_a = data['sentence_a']
#         sentence_b = data['sentence_b']

        
#         model_inputs = tokenizer(sentence_a, sentence_b, return_tensors = 'pt', max_length = 512, truncation=True, padding='max_length')


        

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
