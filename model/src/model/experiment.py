import math
import time
import random
import os
import argparse
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertForNextSentencePrediction
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from .model import BERTModel, Seq2Seq
from . import transforms
from .config import cfg
from .dataset import CovidDataset
from datasets import load_metric



def calc_accuracy(output,Y,mask):
    """
    Calculate the accuracy (point by point evaluation)
    :param output: output from the model (tensor)
    :param Y: ground truth given by dataset (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: accuracy used for validation logs (float)
    """
    _ , max_indices = torch.max(output.data,1)
    max_indices = max_indices.view(mask.shape[1], mask.shape[0]).permute(1,0)
    Y = Y.view(mask.shape[1], mask.shape[0]).permute(1,0)
    max_indices = torch.masked_select(max_indices, mask)
    Y = torch.masked_select(Y, mask)
    train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
    return train_acc, max_indices, Y

def loss_function(trg, output, mask):
    """
    Calculate the loss (point by point evaluation)
    :param trg: ground truth given by dataset (tensor)
    :param output: output from the model (tensor)
    :param mask: used to mask out the padding (tensor)
    :return: loss needed for backpropagation and logging (float)
    """
    trg = trg[1:].permute(1,0,2)
    output = output[1:].permute(1,0,2)
    mask = mask.unsqueeze(2).expand(trg.size())
    trg = torch.masked_select(trg, mask)
    output = torch.masked_select(output, mask)
    label_mask = (trg != 0)
    selected = torch.masked_select(output, label_mask)
    loss = -torch.sum(selected) / selected.size()[0]
    return loss

def default_collate(batch,y_padding_value,mode3_padding_value,callsign_padding_value):
    """
    Stack the tensors from dataloader and pad sequences in batch
    :param batch: batch from the torch dataloader
    :return: stacked input to the seq2seq model
    """
    batch.sort(key=lambda x: x[-1], reverse=True)
    batch_x, batch_y, batch_mode3, batch_callsign, batch_len = zip(*batch)
    batch_pad_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first=True)
    batch_pad_y = torch.nn.utils.rnn.pad_sequence(batch_y,batch_first=True,padding_value=y_padding_value)
    batch_pad_mode3 = torch.nn.utils.rnn.pad_sequence(batch_mode3,batch_first=True,padding_value=mode3_padding_value)
    batch_pad_callsign = torch.nn.utils.rnn.pad_sequence(batch_callsign,batch_first=True,padding_value=callsign_padding_value)

    batch_len = torch.Tensor(batch_len).type(torch.int64)
    return [batch_pad_x, batch_pad_y, batch_pad_mode3, batch_pad_callsign, batch_len]



class Experiment(object):
   #should init as arguments here 
    def __init__(self, args, clearml_task=None):
        
        self.clearml_task = clearml_task
        self.datapath = args.data_path
        self.features = args.data_features
        self.callsign_column = args.data_identifiers_callsign_data_column
        self.mode3_column = args.data_identifiers_mode3_data_column
        self.time_encoding_dims = args.data_time_encoding_dims
        self.n_features = (len(args.data_features) + self.time_encoding_dims -1) if self.time_encoding_dims else len(args.data_features)
        self.label = args.data_label
       
        self.weight_by = args.data_weight_by

        
        self.hid_dim = args.model_hidden_size
        self.n_layers = args.model_hidden_layers
        self.enc_dropout = args.model_enc_dropout
        self.dec_dropout = args.model_dec_dropout
        self.teacher_forcing = args.model_teacher_forcing


        self.checkpoint_dir = args.train_checkpoint_dir
        self.batch_size = args.train_batch_size
        self.learning_rate = args.train_lr

        
        self.auto_lr = args.train_auto_lr
        self.n_gpu = args.train_n_gpu
        self.accelerator = args.train_accelerator
        self.model_save_period = args.train_model_save_period
        self.log_every_n_steps = args.train_log_every_n_steps
        self.save_top_k = args.train_save_top_k
        self.num_workers = args.train_num_workers

        self.n_mode3_token_embedding = args.model_n_mode3_token_embedding 
        self.n_mode3_token_layers = args.model_n_mode3_token_layers

        self.n_callsign_token_embedding = args.model_n_callsign_token_embedding 
        self.n_callsign_token_layers = args.model_n_callsign_token_layers
        
        self.seed = args.train_seed
        self.transforms = cfg['data']['transforms']
        self.lr_schedule = cfg['train']['lr_schedule']

        self.use_uncased = args.model_use_uncased
        self.max_length = args.model_sequence_length
        self.n_epochs = args.train_num_epochs
        


    def _get_logger(self):
        logger = TensorBoardLogger(self.checkpoint_dir, name='logs')
        return logger
    def _get_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename = '{k}-{epoch}',
            save_top_k= self.save_top_k,
            verbose=True,
            monitor='val_loss',
            mode='min',
            period = self.model_save_period
            )
        lr_logging_callback = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback,lr_logging_callback]
        return callbacks

    def run_experiment(self, task:str, round:int):
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)

        pl.seed_everything(self.seed)

        ##### new #####
        train_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="train", max_length=self.max_length)
        valid_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="valid", max_length=self.max_length)
        test_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="test", max_length=self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        total_training_steps = self.n_epochs*self.batch_size
        # take a fifth of training steps for warmup
        warmup_steps = total_training_steps//5
        model = BERTModel(use_uncased=self.use_uncased, task=task, round=round, lr=self.learning_rate, num_training_steps=total_training_steps, num_warmup_steps=warmup_steps)

        callbacks = self._get_callbacks()
        logger = self._get_logger()
        
        trainer = pl.Trainer(
            gpus=self.n_gpu,
            accelerator=self.accelerator if self.n_gpu > 1 else None,
            callbacks=callbacks,
            logger=logger,
            max_epochs=self.n_epochs,
            default_root_dir = self.checkpoint_dir,
            log_every_n_steps=self.log_every_n_steps
        )

        if self.auto_lr:
            lr_finder = trainer.tuner.lr_find(model,train_loader,valid_loader)
            new_lr = lr_finder.suggestion()
            model.learning_rate = new_lr
        
        trainer.fit(model, train_loader, valid_loader)

        # train_dataset = FlightDataset(self.datapath,self.features,self.label,self.mode3_column,self.callsign_column,"train",self.transforms,self.time_encoding_dims)
        # valid_dataset = FlightDataset(self.datapath,self.features,self.label,self.mode3_column,self.callsign_column,"valid",self.transforms,self.time_encoding_dims)

        # y_padding = train_dataset.labels_map['pad']
        # callsign_padding = train_dataset.CALLSIGN_CHAR2IDX['_']
        # mode3_padding = train_dataset.MODE3_CHAR2IDX['_']
        # train_loader = DataLoader(train_dataset, collate_fn=lambda x: default_collate(x,y_padding,mode3_padding,callsign_padding),\
        #     batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # valid_loader = DataLoader(valid_dataset, collate_fn=lambda x: default_collate(x,y_padding,mode3_padding,callsign_padding),\
        #     batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        

        # class_weights = {
        #     'label_segment_count': train_dataset.get_class_weights('label_segment_count'),
        #     'label_point_count': train_dataset.get_class_weights('label_point_count')
        # }
        # for batch in train_loader:
        #     print(batch[0].shape)
        #     print(batch[1])
        #     # print(batch[2])
        #     break
        # -2 for n_class because we have two special tokens

        # labels_map = train_dataset.labels_map
        # n_callsign_tokens = len(train_dataset.CALLSIGN_CHAR2IDX)
        # n_mode3_tokens = len(train_dataset.MODE3_CHAR2IDX)
        # n_classes = train_dataset.n_classes
        # distributed = self.n_gpu > 1
        # if self.clearml_task:
        #     self.clearml_task.connect_configuration({str(i):val for i,val in enumerate(class_weights[self.weight_by].cpu().numpy())},name='Class Weights')
        #     self.clearml_task.connect_configuration(labels_map,name='Labels Map')

        #     metas = {'Train':train_dataset.metadata.copy(),'Valid':valid_dataset.metadata.copy()}
        #     for meta in metas.keys():
        #         for key in ['labels','length','track_ids']:
        #             metas[meta].pop(key)
        #         self.clearml_task.connect_configuration(metas[meta],name='{} Metadata'.format(meta))

        
        # Run training and get the model type and case it was training on
        # model_type, model_case = model()
        # # get saved model
        # model_name = "round" + str(round) + "_model"
        # if model_type == "BertForNextSentencePrediction":
        #     curr_model = BertForNextSentencePrediction.from_pretrained(model_case, state_dict=torch.load(model_name))
        # elif model_type == "BertForMaskedLM":
        #     curr_model = BertForMaskedLM.from_pretrained(model_case, state_dict=torch.load(model_name))
        
        # # calc accuracy on test data
        # calc_accuracy(model=curr_model, test_loader=test_loader, device=device)

        

        # model = Seq2Seq(self.learning_rate, self.lr_schedule, self.hid_dim, self.n_layers, self.n_features,\
        #     self.enc_dropout, self.dec_dropout, n_mode3_tokens,self.n_mode3_token_embedding, self.n_mode3_token_layers, n_callsign_tokens, self.n_callsign_token_embedding, self.n_callsign_token_layers,\
        #     n_classes ,self.teacher_forcing,class_weights,self.weight_by,\
        #     labels_map,distributed)
       
        
    
    @staticmethod
    def add_experiment_args(parent_parser):

        def get_unnested_dict(d,root=''):
            unnested_dict = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    unnested_dict.update(get_unnested_dict(value,root+key+'_'))
                else:
                    unnested_dict[root+key]=value
            return unnested_dict
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        unnested_args = get_unnested_dict(cfg)
        for key,value in unnested_args.items():
            #do not parse transforms and lr schedule as we want them as nested dicts
            if 'transforms' not in key and 'lr_schedule' not in key:
                parser.add_argument('--'+key,default=value)

        return parser

    @staticmethod
    def create_torchscript_model(model_name):
        model = Seq2Seq.load_from_checkpoint(os.path.join(cfg['train']['checkpoint_dir'],model_name))
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(cfg['train']['checkpoint_dir'],"model.pt"))
    
