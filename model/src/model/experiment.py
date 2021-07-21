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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers.utils.dummy_pt_objects import MPNET_PRETRAINED_MODEL_ARCHIVE_LIST

from .model import BERTModel
from .config import cfg
from .dataset import CovidDataset
from .transforms import Transformations

from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

class Experiment(object):
   #should init as arguments here 
    def __init__(self, args):
        self.datapath = args.data_path

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
        
        self.seed = args.train_seed

        self.use_uncased = args.model_use_uncased
        self.max_length = args.model_sequence_length
        self.n_epochs = args.train_num_epochs

    def _get_logger(self):
        logger = TensorBoardLogger(self.checkpoint_dir, name='logs')
        return logger
    def _get_callbacks(self, model_name):
        checkpoint_callback = CustomCheckpoint(
            dirpath=self.checkpoint_dir,
            filename = 'SAMPLE-1000-' + model_name + '-{epoch}',
            save_top_k= self.save_top_k,
            verbose=True,
            monitor='val_loss',
            mode='min',
            period = self.model_save_period
            )
        lr_logging_callback = LearningRateMonitor(logging_interval='step')
        if (model_name.upper() == "QA"):
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=3,
                verbose=True,
                mode='min',
                strict=True,
                check_finite=True,
                stopping_threshold=None,
                divergence_threshold=None,
                check_on_train_epoch_end=False
            )
            callbacks = [checkpoint_callback, lr_logging_callback, early_stopping_callback]
        else:
            callbacks = [checkpoint_callback, lr_logging_callback]
        return callbacks

    def run_experiment(self, task:str, model_startpt:str=None):

        pl.seed_everything(self.seed)

        # transform data
        transformation_train = Transformations(task=task, mode="train", use_uncased=self.use_uncased)
        train_length = transformation_train()
        transformation_valid = Transformations(task=task, mode="valid", use_uncased=self.use_uncased)
        valid_length = transformation_valid()
        transformation_test = Transformations(task=task, mode="test", use_uncased=self.use_uncased)
        test_length = transformation_test()

        # load dataset
        train_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="train", max_length=self.max_length, data_length=train_length)
        valid_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="valid", max_length=self.max_length, data_length=valid_length)
        test_dataset = CovidDataset(use_uncased=self.use_uncased, task=task, mode="test", max_length=self.max_length, data_length=test_length)

        # data loaders
        train_loader = DataLoader(dataset = train_dataset, num_workers=self.num_workers, shuffle=True, collate_fn=train_dataset.collate_fn, batch_size=self.batch_size)
        valid_loader = DataLoader(dataset = valid_dataset, num_workers=self.num_workers, shuffle=False, collate_fn=valid_dataset.collate_fn, batch_size=self.batch_size)
        test_loader = DataLoader(dataset = test_dataset, num_workers=self.num_workers, shuffle=False, collate_fn=test_dataset.collate_fn, batch_size=self.batch_size)

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
        # model = model.cuda()
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
    
# Overwrite Pytorch's ModelCheckpoint to save model to S3. If on local machine, just ignore this part
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
    
    def _save_model(self, trainer: 'pl.Trainer', filepath: str) -> None:
        if trainer.training_type_plugin.rpc_enabled:
            # RPCPlugin manages saving all model states
            # TODO: the rpc plugin should wrap trainer.save_checkpoint
            # instead of us having to do it here manually
            trainer.training_type_plugin.rpc_save_model(trainer, self._do_save, filepath)
        else:
            self._do_save(trainer, filepath)

        # call s3 function here to upload file to s3 using filepath

       
        best_model_path = self.best_model_path
        best_model_name = best_model_path.split("/")[-1]
        # call s3 function here to upload the best_model file to s3 
        # using best_model_path (which is the path for the best model) and best_model_name (which is just the name of the best_model file)
        # but instead of doing the s3_upload_file(best_model_path, best_model_path, sth_here) or sth (i forgot the exact func) to save the best_model file to the same
        # path name as the best_model in the local dir, keep saving the best_model file in the same folder for e.g. trained_models/best_models/best_model_name
        # so its sth like s3_upload_file("path/to/src/trained_models/best_models/best_model_name", best_model_path, sth_here)
        # or s3_upload_file(best_model_path, "path/to/src/trained_models/best_model/best_model_name")
