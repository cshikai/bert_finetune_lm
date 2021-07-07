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
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers.utils.dummy_pt_objects import MPNET_PRETRAINED_MODEL_ARCHIVE_LIST

from .model import BERTModel
from .config import cfg
from .dataset import CovidDataset

from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union



# def calc_accuracy(output,Y,mask):
#     """
#     Calculate the accuracy (point by point evaluation)
#     :param output: output from the model (tensor)
#     :param Y: ground truth given by dataset (tensor)
#     :param mask: used to mask out the padding (tensor)
#     :return: accuracy used for validation logs (float)
#     """
#     _ , max_indices = torch.max(output.data,1)
#     max_indices = max_indices.view(mask.shape[1], mask.shape[0]).permute(1,0)
#     Y = Y.view(mask.shape[1], mask.shape[0]).permute(1,0)
#     max_indices = torch.masked_select(max_indices, mask)
#     Y = torch.masked_select(Y, mask)
#     train_acc = (max_indices == Y).sum().item()/max_indices.size()[0]
#     return train_acc, max_indices, Y

# def loss_function(trg, output, mask):
#     """
#     Calculate the loss (point by point evaluation)
#     :param trg: ground truth given by dataset (tensor)
#     :param output: output from the model (tensor)
#     :param mask: used to mask out the padding (tensor)
#     :return: loss needed for backpropagation and logging (float)
#     """
#     trg = trg[1:].permute(1,0,2)
#     output = output[1:].permute(1,0,2)
#     mask = mask.unsqueeze(2).expand(trg.size())
#     trg = torch.masked_select(trg, mask)
#     output = torch.masked_select(output, mask)
#     label_mask = (trg != 0)
#     selected = torch.masked_select(output, label_mask)
#     loss = -torch.sum(selected) / selected.size()[0]
#     return loss

# def default_collate(batch,y_padding_value,mode3_padding_value,callsign_padding_value):
#     """
#     Stack the tensors from dataloader and pad sequences in batch
#     :param batch: batch from the torch dataloader
#     :return: stacked input to the seq2seq model
#     """
#     batch.sort(key=lambda x: x[-1], reverse=True)
#     batch_x, batch_y, batch_mode3, batch_callsign, batch_len = zip(*batch)
#     batch_pad_x = torch.nn.utils.rnn.pad_sequence(batch_x,batch_first=True)
#     batch_pad_y = torch.nn.utils.rnn.pad_sequence(batch_y,batch_first=True,padding_value=y_padding_value)
#     batch_pad_mode3 = torch.nn.utils.rnn.pad_sequence(batch_mode3,batch_first=True,padding_value=mode3_padding_value)
#     batch_pad_callsign = torch.nn.utils.rnn.pad_sequence(batch_callsign,batch_first=True,padding_value=callsign_padding_value)

#     batch_len = torch.Tensor(batch_len).type(torch.int64)
#     return [batch_pad_x, batch_pad_y, batch_pad_mode3, batch_pad_callsign, batch_len]



class Experiment(object):
   #should init as arguments here 
    def __init__(self, args, clearml_task):
        
        self.clearml_task = clearml_task
        self.datapath = args.data_path
        # self.features = args.data_features
        # self.callsign_column = args.data_identifiers_callsign_data_column
        # self.mode3_column = args.data_identifiers_mode3_data_column
        # self.time_encoding_dims = args.data_time_encoding_dims
        # self.n_features = (len(args.data_features) + self.time_encoding_dims -1) if self.time_encoding_dims else len(args.data_features)
        # self.label = args.data_label
       
        # self.weight_by = args.data_weight_by

        
        # self.hid_dim = args.model_hidden_size
        # self.n_layers = args.model_hidden_layers
        # self.enc_dropout = args.model_enc_dropout
        # self.dec_dropout = args.model_dec_dropout
        # self.teacher_forcing = args.model_teacher_forcing


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

        # self.n_mode3_token_embedding = args.model_n_mode3_token_embedding 
        # self.n_mode3_token_layers = args.model_n_mode3_token_layers

        # self.n_callsign_token_embedding = args.model_n_callsign_token_embedding 
        # self.n_callsign_token_layers = args.model_n_callsign_token_layers
        
        self.seed = args.train_seed
        # self.transforms = cfg['data']['transforms']
        # self.lr_schedule = cfg['train']['lr_schedule']

        self.use_uncased = args.model_use_uncased
        self.max_length = args.model_sequence_length
        self.n_epochs = args.train_num_epochs

    def _get_logger(self):
        logger = TensorBoardLogger(self.checkpoint_dir, name='logs')
        return logger
    def _get_callbacks(self, model_name):
        checkpoint_callback = CustomCheckpoint(
            dirpath=self.checkpoint_dir,
            filename = model_name + '-{k}-{epoch}',
            save_top_k= self.save_top_k,
            verbose=True,
            monitor='val_loss',
            mode='min',
            period = self.model_save_period
            )
        lr_logging_callback = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback,lr_logging_callback]
        return callbacks

    def run_experiment(self, task:str, model_startpt:str=None):

        pl.seed_everything(self.seed)

        train_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="train", max_length=self.max_length)
        train_batch_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=self.batch_size, drop_last = True)
        valid_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="valid", max_length=self.max_length)
        valid_batch_sampler = BatchSampler(RandomSampler(valid_dataset), batch_size=self.batch_size, drop_last = True)
        test_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="test", max_length=self.max_length)
        test_batch_sampler = BatchSampler(RandomSampler(test_dataset), batch_size=self.batch_size, drop_last = True)

        train_loader = DataLoader(dataset = train_dataset, batch_sampler = train_batch_sampler, collate_fn=train_dataset.collate_fn, num_workers=self.num_workers)
        valid_loader = DataLoader(dataset = valid_dataset, batch_sampler = valid_batch_sampler, collate_fn=valid_dataset.collate_fn, num_workers=self.num_workers)
        test_loader = DataLoader(dataset = test_dataset, batch_sampler = test_batch_sampler, collate_fn=test_dataset.collate_fn, num_workers=self.num_workers)

        steps_per_epoch = len(train_dataset) // self.batch_size
        total_training_steps = self.n_epochs*steps_per_epoch
        # take a fifth of training steps for warmup
        warmup_steps = total_training_steps//5
        distributed = self.n_gpu > 1

        model = BERTModel(use_uncased=self.use_uncased,
                          task=task,
                          lr=self.learning_rate,
                          num_training_steps=total_training_steps,
                          num_warmup_steps=warmup_steps, 
                          seq_length=self.max_length,
                          distributed=distributed,
                          model_startpt = model_startpt)
        model = model.cuda()
        callbacks = self._get_callbacks(task)
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

        best_model = callbacks[0].best_model_path
        # run test on best model
        trainer.test(test_dataloaders = test_loader, ckpt_path=best_model)
        
        return best_model

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

    # @staticmethod
    # def create_torchscript_model(model_name):
    #     model = Seq2Seq.load_from_checkpoint(os.path.join(cfg['train']['checkpoint_dir'],model_name))
    #     script = model.to_torchscript()
    #     torch.jit.save(script, os.path.join(cfg['train']['checkpoint_dir'],"model.pt"))
    
class CustomCheckpoint(ModelCheckpoint):
    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: Optional[int] = None,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        every_n_val_epochs: Optional[int] = None,
        period: Optional[int] = None,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            every_n_val_epochs,
            period,
        )
        

        # self.monitor = monitor
        # self.verbose = verbose
        # self.save_last = save_last
        # self.save_top_k = save_top_k
        # self.save_weights_only = save_weights_only
        # self.auto_insert_metric_name = auto_insert_metric_name
        # self._last_global_step_saved = -1
        # self._last_time_checked: Optional[float] = None
        # self.current_score = None
        # self.best_k_models = {}
        # self.kth_best_model_path = ""
        # self.best_model_score = None
        # self.best_model_path = ""
        # self.last_model_path = ""

        # self.__init_monitor_mode(mode)
        # self.__init_ckpt_dir(dirpath, filename, save_top_k)
        # self.__init_triggers(every_n_train_steps, every_n_val_epochs, train_time_interval, period)
        # self.__validate_init_configuration()
        # self._save_function = None

    def _save_model(self, trainer: 'pl.Trainer', filepath: str) -> None:
        if trainer.training_type_plugin.rpc_enabled:
            # RPCPlugin manages saving all model states
            # TODO: the rpc plugin should wrap trainer.save_checkpoint
            # instead of us having to do it here manually
            trainer.training_type_plugin.rpc_save_model(trainer, self._do_save, filepath)
        else:
            self._do_save(trainer, filepath)

        # call s3 function here to upload file to s3 using filepath
        