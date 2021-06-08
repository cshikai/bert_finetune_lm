from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
import os
import numpy as np


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
