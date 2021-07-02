from typing import List, Callable
import os
import json
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
# from datasets import Dataset
from torchvision.transforms import Compose
from transformers import BertTokenizer, BertTokenizerFast
#import dask.dataframe as dd

from .transforms import TimeEncoder #TODO change this to importing our methods in transforms.py
from . import transforms
from typing import Dict



class CovidDataset(Dataset):
    """
    Covid_Dataset Object
    """
    def __init__(self, use_uncased:bool, task:str, mode:str, max_length:int):
        self.use_uncased = use_uncased
        self.task = task
        self.mode = mode
        self.max_length = max_length
        self.task = task 

        if task == "QA":
            # path for QA
            path = 'pipeline/uncased_qna.json' if self.use_uncased else 'pipeline/cased_qna.json'
        elif task == "NSP" or task == "MLM":
            # paths for nsp and mlm
            path = 'pipeline/uncased.json' if self.use_uncased else 'pipeline/cased.json'
        # print("dataset.py: open data file in pipeline folder")
        with open(path) as f:
            all_data = json.load(f)
            self.data_loaded = all_data[self.mode]
        # print("dataset.py: initialise tokenizer class")

        transformation = transforms.Transformations(data=self.data_loaded, task=self.task)
        # print("dataset.py: tokenizing")
        self.data_transformed = transformation()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') if self.use_uncased else BertTokenizerFast.from_pretrained('bert-base-cased')
        
        
        
    def __getitem__(self, batch_idx):
        # tokenize data
        data_tokenized = self.tokenize_steps(batch_idx)

        return data_tokenized


    def __len__(self):
        if self.task == "NSP":
            return len(self.data_transformed['sentence_a'])
        elif self.task == "MLM":
            return len(self.data_transformed['sentence_list'])
        elif self.task == "QA":
            pass
        

    # Choose tokenizing steps based on task
    def tokenize_steps(self, batch_idx):
        if self.task == "NSP":
            data_tokenized = self.tokenize_nsp(batch_idx)
        elif self.task == "MLM":
            data_tokenized = self.tokenize_mlm(batch_idx)
        elif self.task == "QA": 
            data_tokenized = self.tokenize_qna(batch_idx)

        return data_tokenized
        
    # Tokenize for NSP
    def tokenize_nsp(self, batch_idx):
        # tokenize
        data_tokenized = self.tokenizer(self.data_transformed['sentence_a'][batch_idx], self.data_transformed['sentence_b'][batch_idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # post tokenize
        data_tokenized['labels'] = torch.LongTensor([self.data_transformed['labels'][batch_idx]]).T
        return data_tokenized

     # Tokenize for MLM
    def tokenize_mlm(self, batch_idx):
        # tokenize
        data_tokenized = self.tokenizer(self.data_transformed['sentence_list'][batch_idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # post tokenize
        # get labels
        data_tokenized['labels'] = data_tokenized.input_ids.detach().clone()
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
        

     # Tokenize for QNA
    def tokenize_qna(self, batch_idx):
        # tokenize
        data_tokenized = self.tokenizer(self.data_transformed['contexts'][batch_idx], self.data_transformed['questions'][batch_idx], return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        # post tokenize
        start_positions = []
        end_positions = []
        # change char indexing to token indexing
        for i, ans in enumerate(self.data_transformed['answers'][batch_idx]):
            start_positions.append(data_tokenized.char_to_token(i, self.data_transformed['answers'][i]['answer_start'][batch_idx]))
            end_positions.append(data_tokenized.char_to_token(i, self.data_transformed['answers'][i]['answer_end'][batch_idx] - 1))

            # if start position is None, the answer passage has been truncated
            if (start_positions[-1] is None):
                start_positions[-1] = self.max_length
            if (end_positions[-1] is None):
                end_positions[-1] = self.max_length

        data_tokenized.update({'start_positions': start_positions, 'end_positions': end_positions})
        return data_tokenized

    @staticmethod
    def collate_fn(batch):

        return {k: torch.tensor(v) for k,v in batch[0].items()}

    # other methods

# class FlightDataset(Dataset):
#     """
#     Flight_Dataset Object
#     """


#     CALLSIGN_TOKENS = '0123456789abcdefghijklmnopqrstuvwxyz-_' #note that _ is padding to reach charlimit and -  is unknown  
#     CALLSIGN_CHAR2IDX = {v:i for i,v in enumerate(CALLSIGN_TOKENS)}

#     MODE3_TOKENS = '0123456789-_' #note that _ is padding to reach charlimit and -  is unknown  
#     MODE3_CHAR2IDX = {v:i for i,v in enumerate(MODE3_TOKENS)}
    
#     def _map_callsign(self,string):
#         return [self.CALLSIGN_CHAR2IDX[i] for i in string]

#     def _map_mode3(self,string):
#         return [self.MODE3_CHAR2IDX[i] for i in string]

#     def __init__(self, datapath: str, features: List[str], label: str, mode3_column: str, callsign_column: str, mode: str, transforms_dict: Dict[str,List[int]], time_encoding_dims: int) -> None:
#         """
#         Initialises the Flight_Dataset object.
#         :param dataset: numpy array of the dataset (in function_class.py)
#         :param data_labels: labels of the dataset given
#         :param labels_dct: dictionary containing encoding of classes
#         :param mode: train or valid or test; assign encodings of labels
#         """


        
#         self.root_folder = os.path.join(datapath,mode) 
#         with open(os.path.join(self.root_folder,'metadata.json')) as infile:
#             self.metadata = json.load(infile)
#         self.features = features
#         self.features_no_datetime = copy.deepcopy(features)
#         if 'datetime' in features:
#             self.features_no_datetime.remove('datetime')
#         assert 'datetime' in self.features
#         assert 'datetime' not in self.features_no_datetime
#         self.label = label
#         self.labels_map = {l : i for i,l in enumerate(self.metadata['labels'])}
#         self.n_classes = len(self.metadata['labels'])
#         self.labels_map['start'] = len(self.metadata['labels'])
#         self.labels_map['pad'] =  len(self.metadata['labels']) + 1

#         self.callsign_column = callsign_column
#         self.mode3_column = mode3_column
#         # self.labels_map['unknown'] = 2 
#         self.mode = mode
#         self.data = dd.read_parquet(os.path.join(self.root_folder,'data.parquet'),\
#             columns=self.features+[self.label] +[self.callsign_column,self.mode3_column],\
#             engine='fastparquet') #this is lazy loading, its not actually loading into memory
#         transforms_list = transforms.get_transforms(transforms_dict)
#         self.transforms =  Compose(transforms_list)
#         self.length_mapper = {}
#         self.idx_to_track_id = {}
#         idx = 0
#         for k,v in self.metadata['length'].items():
#             int_key = int(k)
#             self.length_mapper[int_key] = v
#             self.idx_to_track_id[idx] = int_key
#             idx = idx + 1

#         self.time_encoding_dims = time_encoding_dims
#         if self.time_encoding_dims:
#             self.time_encoder = TimeEncoder(time_encoding_dims)
        



#     def get_class_weights(self,count_type):
#         '''
#         Method to get weights for each label class, on the premise that we balance the dataset out.
#         count_type : 'label_segment_count' - balances the data segments , 'label_point_count' - balances the datapoints
#         '''

#         counts = self.metadata[count_type]
#         weights = torch.ones(len(counts))
#         #note that weights[0:3] will be one, but that is ok as there will never be actual labels that are 0,1 or 2 (special tokens) 
#         for label,count in counts.items():
#             weights[self.labels_map[label]] = 1/count
#         weights = weights/torch.min(weights)
#         return weights 

#     def __len__(self):
#         '''
#         Get the length of dataset.
#         '''
#         return len(self.metadata['track_ids'])

#     def __getitem__(self, index):
#         '''
#         Get the item for each batch
#         :return: a tuple of 6 object:
#         1) normalized features of dataset
#         2) labels of dataset (one-hot encoded and labels_dct)
#         3) labels of dataset (encoded with labels_dct)
#         4) length of each sequences without padding
#         5) track id of each row in the dataset
#         '''
#         track_id = self.idx_to_track_id[index]
#         data_slice = self.data.loc[track_id].compute().assign(datetime=lambda x: (x.datetime-x.datetime.min()).dt.total_seconds())

    
#         if self.time_encoding_dims:
#             x = torch.from_numpy(data_slice[self.features_no_datetime].values)
#             x = self.transforms(x)
#             x_time = torch.from_numpy(data_slice['datetime'].values)
#             x_time_encoded = self.time_encoder(x_time)
#             x = torch.cat([x,x_time_encoded],-1)


#         else:
#             x = torch.from_numpy(data_slice[self.features].values)
#             x = self.transforms(x)
#         y = torch.from_numpy(data_slice[self.label].map(self.labels_map).astype(int).values)
#         # labels, label_index = preprocess_y(self.data_labels[index], self.labels_dct, self.mode)

#         callsign_string = np.array(data_slice[self.callsign_column].map(self._map_callsign).tolist())
#         callsign_index = torch.from_numpy(callsign_string)
        
#         mode3_string = np.array(data_slice[self.mode3_column].map(self._map_mode3).tolist())
#         mode3_index = torch.from_numpy(mode3_string)
#         # print('x',x.shape)
#         # print('y',y.shape)
#         # print('len',self.length_mapper[track_id])
#         # make sure that the sequence length is the last to be returned for batch collation
#         return x, y, mode3_index, callsign_index, self.length_mapper[track_id]
