import argparse
import random
import numpy as np

import torch
from torch._C import device
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertForQuestionAnswering, BertTokenizerFast, AdamW, get_scheduler
from datasets import load_metric
from torchmetrics import Accuracy, F1
import torch.nn.functional as F

from .config import cfg
from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention
from .id_encoder import IdEncoder
from . import results


class BERTModel(pl.LightningModule):
    def __init__(self, use_uncased:bool, task:str, lr:float, num_training_steps, num_warmup_steps, seq_length: int, distributed: bool, model_startpt:str = None):
        super().__init__()
        self.use_uncased = use_uncased
        self.task = task.upper()
        # self.batch_size = batch_size
        self.seq_length = seq_length
        # self.train_dataloader = train_dataloader
        # self.eval_dataloader = eval_dataloader
        # self.num_epochs = num_epochs
        
        
        self.model_startpoint =model_startpt
        self.bert_case_uncase = 'bert-base-uncased' if self.use_uncased else 'bert-base-cased'
        # declare model and other stuff like optimizers here
        # start training the model from fresh pre-trained BERT
        if (self.model_startpoint is None):
            if (self.task == "NSP"):
                self.bert = BertForNextSentencePrediction.from_pretrained(self.bert_case_uncase)
            elif (self.task == "MLM"):
                self.bert = BertForMaskedLM.from_pretrained(self.bert_case_uncase)
            elif (self.task == "QA"):
                self.bert = BertForQuestionAnswering.from_pretrained(self.bert_case_uncase)
                self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_case_uncase)
        # start training the model from previously trained model which was saved
        else:
            if (self.task == "NSP"):
                self.bert = BertForNextSentencePrediction.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
                    # self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', state_dict=torch.load(self.model_startpoint, map_location='cpu')) #to load on cpu
            elif (self.task == "MLM"):
                self.bert = BertForMaskedLM.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
            elif (self.task == "QA"):
                self.bert = BertForQuestionAnswering.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
                self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_case_uncase)

        # self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        # self.lr_scheduler = get_scheduler('linear', optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps)
        # self.maxAccuracy = -1

        self.lr = lr
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        
        self.distributed = distributed
        #loss
        # self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)

        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

    # metric for NSP
    def calculate_accuracy(self, logits, labels):
        NSPpredictions = torch.argmax(logits, dim=-1)
        NSPtarget = torch.reshape(labels, (-1,))
        accuracy = Accuracy().to(device="cuda")
        return accuracy(NSPpredictions, NSPtarget)

        
    # metric for MLM
    def calculate_perplexity(self, logits, labels):
        first_dim = list(logits.shape)[0] * self.seq_length
        MLMpredictions = torch.reshape(logits, (first_dim, -1)) #reshape tensor to (batch_size*seq_length, vocab_size)
        MLMtarget = torch.reshape(labels, (-1,)) #reshape tensor to (batch_size*seq_length)
        loss = nn.functional.cross_entropy(MLMpredictions, MLMtarget)
        perplexity = torch.exp(loss)
        return perplexity

    # metrics for QA
    # function to get the actual answers for a batch of input
    def get_actual_answers(self, input_ids, start_positions, end_positions):
        answers = []
        for i, seq in enumerate(input_ids):
            ids = input_ids[i][start_positions[i]:end_positions[i]+1]
            answers.append(self.tokenizer.decode(ids))
        return answers
    # function to get the predicted answers for a batch of input
    def get_pred_answers(self, input_ids, start_logits, end_logits):
        answers = []
        for i, seq in enumerate(input_ids):
            start_position = torch.argmax(start_logits[i])
            end_position = torch.argmax(end_logits[i][start_position:]) + start_position
            ids = input_ids[i][start_position:end_position+1]
            answers.append(self.tokenizer.decode(ids))
        return answers

    # F1 Score
    def calculate_f1(self, input_ids, start_positions, end_positions, start_logits, end_logits):
        actual_ans = self.get_actual_answers(input_ids, start_positions, end_positions)
        pred_ans = self.get_pred_answers(input_ids, start_logits, end_logits)
        f1 = 0
        length = len(actual_ans)
        for i in range(length):
            # if either one ans is blank
            if len(actual_ans[i]) == 0 or len(pred_ans[i]) == 0:
                # if both are blank, it will return 1. Else 0.
                f1 += int(actual_ans[i] == pred_ans[i])
            else:
                # turn ans into a set of strings
                actual_ans_set = set(actual_ans[i].split(" "))
                pred_ans_set = set(pred_ans[i].split(" "))

                # get set of common words
                common_words = actual_ans_set & pred_ans_set

                # if no common words, f1 = 1
                if len(common_words) == 0:
                    f1 += 0 
                # if there are some common words, calculate f1
                else: 
                    prec = len(common_words) / len(pred_ans_set)
                    rec = len(common_words) / len(actual_ans_set)
                    f1 += (2 * ((prec * rec) / (prec + rec)))

        f1 /= length
        return f1
    
    # Exact Match
    def calculate_exactmatch(self, input_ids, start_positions, end_positions, start_logits, end_logits):
        actual_ans = self.get_actual_answers(input_ids, start_positions, end_positions)
        pred_ans = self.get_pred_answers(input_ids, start_logits, end_logits)
        em = 0
        length = len(actual_ans)
        for i in range(length):
            em += int(actual_ans[i]==pred_ans[i])
        em /= length
        return em

    def forward(self, input_ids, attention_mask, labels, start_positions, end_positions, token_type_ids):
        """
        Forward propagation of one batch.
        """
        if (self.task == "QA"):
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        else:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids)

        return output

    def training_step(self, batch, batch_idx):
        """
        Pytorch lightning training step.
        """
        # calls forward, loss function, accuracy function, perplexity function
        # decide what happens to one batch of data here

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = None
        labels = None
        start_positions = None
        end_positions = None
        if (self.task == "QA"):
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
        elif self.task == "NSP":
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
        else:
            labels = batch['labels']

        # call forward
        output = self(input_ids, attention_mask, labels, start_positions, end_positions, token_type_ids)
        loss = output.loss
        
        # log metrices
        self.log('train_loss', loss, sync_dist=self.distributed)
        if (self.task == "NSP"):
            accuracy = self.calculate_accuracy(output.logits, labels)
            self.log('train_acc', accuracy, sync_dist=self.distributed)
        elif (self.task == "MLM"):
            perplexity = self.calculate_perplexity(output.logits, labels)
            self.log('train_perplex', perplexity, sync_dist=self.distributed)
        elif (self.task == "QA"):
            em = self.calculate_exactmatch(input_ids, start_positions, end_positions, output.start_logits, output.end_logits)
            self.log('train_exactmatch', em, sync_dist=self.distributed)
            f1 = self.calculate_f1(input_ids, start_positions, end_positions, output.start_logits, output.end_logits)
            self.log('train_f1', f1, sync_dist=self.distributed)

        
        # return {"loss": loss, "predictions": output, "labels": labels}
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Pytorch lightning validation step.
        """
        # our evaluate function but for one batch and without the code to decide best epoch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = None
        labels = None
        start_positions = None
        end_positions = None
        if (self.task == "QA"):
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
        elif self.task == "NSP":
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
        else:
            labels = batch['labels']

        # call forward
        output = self(input_ids, attention_mask, labels, start_positions, end_positions, token_type_ids)
        loss = output.loss

        # log metrices
        accuracy = 0
        perplexity = 0
        em = 0
        self.log('val_loss', loss, sync_dist=self.distributed)
        if (self.task == "NSP"):
            accuracy = self.calculate_accuracy(output.logits, labels)
            self.log('val_acc', accuracy, sync_dist=self.distributed)
        elif (self.task == "MLM"):
            perplexity = self.calculate_perplexity(output.logits, labels)
            self.log('val_perplex', perplexity, sync_dist=self.distributed)
        elif (self.task == "QA"):
            em = self.calculate_exactmatch(input_ids, start_positions, end_positions, output.start_logits, output.end_logits)
            self.log('val_exactmatch', em, sync_dist=self.distributed)
            f1 = self.calculate_f1(input_ids, start_positions, end_positions, output.start_logits, output.end_logits)
            self.log('val_f1', f1, sync_dist=self.distributed)

        return {
            'val_loss': loss,
            'val_acc': accuracy,
            'val_perplex': perplexity,
            'val_em': em,
            # 'labels':y,
            # 'predictions':max_indices,
            # 'confidence':confidence,
            # 'seg_labels': seg_y,
            # 'seg_predictions': seg_pred
            }
    
    def test_step(self, batch, batch_idx):
        """
        Pytorch lightning validation step.
        """
        # our evaluate function but for one batch and without the code to decide best epoch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = None
        labels = None
        start_positions = None
        end_positions = None
        if (self.task == "QA"):
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
        elif self.task == "NSP":
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
        else:
            labels = batch['labels']

        # call forward
        output = self(input_ids, attention_mask, labels, start_positions, end_positions, token_type_ids)
        loss = output.loss
        
        # log metrices
        accuracy = 0
        perplexity = 0
        em = 0
        self.log('test_loss', loss, sync_dist=self.distributed)
        if (self.task == "NSP"):
            accuracy = self.calculate_accuracy(output.logits, labels)
            self.log('test_acc', accuracy, sync_dist=self.distributed)
        elif (self.task == "MLM"):
            perplexity = self.calculate_perplexity(output.logits, labels)
            self.log('test_perplex', perplexity, sync_dist=self.distributed)
        elif (self.task == "QA"):
            em = self.calculate_exactmatch(input_ids, start_positions, end_positions, output.start_logits, output.end_logits)
            self.log('test_exactmatch', em, sync_dist=self.distributed)

        return {
            'test_loss': loss,
            'test_acc': accuracy,
            'test_perplex': perplexity,
            'test_em': em,
            # 'labels':y,
            # 'predictions':max_indices,
            # 'confidence':confidence,
            # 'seg_labels': seg_y,
            # 'seg_predictions': seg_pred
            }

    def validation_epoch_end(self, output):
        pass
