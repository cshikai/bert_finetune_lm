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



class PMCDataPipeline(object):

    def __init__(self):
        pass
        self.use_lower_only 
    def __call__(self):
        #your pipeline code here
        if use_lower_only:
            #do sth
        else:
            #do else
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
