import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertForNextSentencePrediction, AdamW, get_scheduler
from datasets import load_metric
from torchmetrics import Accuracy
import torch.nn.functional as F

from .config import cfg
from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention
from .id_encoder import IdEncoder
from . import results

from transformers import BertForNextSentencePrediction, BertForMaskedLM

class BERTModel(pl.LightningModule):
    def __init__(self, use_uncased:bool, task:str, round:int, lr:float, num_training_steps, num_warmup_steps, seq_length: int, distributed: bool):
        super().__init__()
        self.use_uncased = use_uncased
        self.task = task.upper()
        self.round = round
        # self.batch_size = batch_size
        self.seq_length = seq_length
        # self.train_dataloader = train_dataloader
        # self.eval_dataloader = eval_dataloader
        # self.num_epochs = num_epochs
        
        
        self.model_startpoint = "round" + str(self.round-1) + "_model"
        self.bert_case_uncase = 'bert-base-uncased' if self.use_uncased else 'bert-base-cased'
        # declare model and other stuff like optimizers here
        # start training the model from fresh pre-trained BERT
        if (self.round == 1):
            if (self.task == "NSP"):
                self.bert = BertForNextSentencePrediction.from_pretrained(self.bert_case_uncase)
            elif (self.task == "MLM"):
                self.bert = BertForMaskedLM.from_pretrained(self.bert_case_uncase)
        # start training the model from previously trained model which was saved
        elif (self.round > 1):
            if (self.task == "NSP"):
                self.bert = BertForNextSentencePrediction.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
                    # self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', state_dict=torch.load(self.model_startpoint, map_location='cpu')) #to load on cpu
            elif (self.task == "MLM"):
                self.bert = BertForMaskedLM.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))

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

    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward propagation of one batch.
        """
        print("model.py: forward")
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        
        return loss, output #not sure what to return...

    def training_step(self, batch, batch_idx):
        """
        Pytorch lightning training step.
        """
        print("model.py: training_step")
        # calls forward, loss function, accuracy function, perplexity function
        # decide what happens to one batch of data here
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # call forward
        loss, output = self(input_ids, attention_mask, labels)
        
        # log metrices
        self.log('train_loss', loss, sync_dist=self.distributed)
        if (self.task == "NSP"):
            NSPpredictions = torch.argmax(output.logits, dim=-1)
            NSPactual = torch.reshape(labels, (-1,))
            accuracy = self.calculate_accuracy(NSPpredictions, NSPactual) #not sure about the paras
            self.log('train_acc', accuracy, sync_dist=self.distributed)
            # print("NSPpredictions: ", NSPpredictions)
            # print("NSPactual: ", NSPactual)
            # print("train accuracy: ", accuracy)
        elif (self.task == "MLM"):
            print("train output.logits: ", output.logits.shape)
            # print("train output.logits: ", output.logits)
            # print("train labels: ", labels.shape)
            # print("train labels: ", labels)
            # MLMinput = output.logits[:, -1, :]
            first_dim = list(output.logits.shape)[0] * self.seq_length
            MLMinput = torch.reshape(output.logits, (first_dim, -1)) #reshape tensor to (batch_size*seq_length, vocab_size)
            # print("train MLMinput: ", MLMinput.shape)
            # print("train MLMinpit: ", MLMinput)
            MLMtarget = torch.reshape(labels, (-1,)) #reshape tensor to (batch_size*seq_length)
            # print("train MLMtarget: ", MLMtarget.shape)
            # print("train MLMactual: ", MLMtarget)
            perplexity = self.calculate_perplexity(MLMinput, MLMtarget) #TODO check if paras are okay, esp after reshaping them
            print("train perplex: ", perplexity)
            # perplexity = self.calculate_perplexity(output.logits, labels) #not sure about the paras
            self.log('train_perplex', perplexity, sync_dist=self.distributed)
        
        return {"loss": loss, "predictions": output, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """
        Pytorch lightning validation step.
        """
        print("model.py: val_step")
        # our evaluate function but for one batch and without the code to decide best epoch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # call forward
        loss, output = self(input_ids, attention_mask, labels)
        # _, max_indices = torch.max(output,1) #idk if we need this
        # log metrices
        self.log('val_loss', loss, sync_dist=self.distributed)
        if (self.task == "NSP"):
            NSPpredictions = torch.argmax(output.logits, dim=-1)
            NSPactual = torch.reshape(labels, (-1,))
            accuracy = self.calculate_accuracy(NSPpredictions, NSPactual) #not sure about the paras
            self.log('val_acc', accuracy, sync_dist=self.distributed)
        elif (self.task == "MLM"):
            # print("val output.logits: ", output.logits.shape)
            # print("val output.logits: ", output.logits)
            # print("val labels: ", labels.shape)
            # print("val labels: ", labels)
            # MLMinput = output.logits[:, -1, :]
            first_dim = list(output.logits.shape)[0] * self.seq_length
            MLMinput = torch.reshape(output.logits, (first_dim, -1)) #reshape tensor to (batch_size*seq_length, vocab_size)
            # print("val MLMinput: ", MLMinput.shape)
            # print("val MLMinpit: ", MLMinput)
            MLMtarget = torch.reshape(labels, (-1,)) #reshape tensor to (batch_size*seq_length)
            # print("val MLMtarget: ", MLMtarget.shape)
            # print("val MLMactual: ", MLMtarget)
            perplexity = self.calculate_perplexity(MLMinput, MLMtarget) #TODO check if paras are okay, esp after reshaping them
            print("val perplex: ", perplexity)
            self.log('val_perplex', perplexity, sync_dist=self.distributed)

        return loss
        # {
        #     'val_loss': loss,
        #     'val_acc': accuracy,
        #     'val_perplex': perplexity,
        #     # 'labels':y,
        #     # 'predictions':max_indices,
        #     # 'confidence':confidence,
        #     # 'seg_labels': seg_y,
        #     # 'seg_predictions': seg_pred
        #     }

    def configure_optimizers(self):
        print("model.py: configure_optimizers")
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)

        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

    # metric for NSP
    def calculate_accuracy(self, output, target):
        print("model.py: calc acc")
        accuracy = Accuracy()
        output = output.to(device="cpu")
        target = target.to(device="cpu")
        # print(accuracy(output,target).device)
        return accuracy(output, target)
        
    # metric for MLM
    def calculate_perplexity(self, output, target):
        print("model.py: calc perplex")
        output = output.to(device="cpu")
        target = target.to(device="cpu")
        loss = nn.functional.cross_entropy(output, target)
        perplexity = torch.exp(loss)
        return perplexity

    # # dont need call function anymore bc got forward())
    # def __call__(self):
    #     self.model.to(self.device)
    #     self.trainingLoop()
    #     return self.model_type, self.bert_case_uncase

    # def trainingLoop(self):
    #     # start training
    #     for epoch in range(self.num_epochs):
    #         self.model.train()
    #         for batch in self.train_dataloader:
    #             batch = {k : v.to(self.device) for k, v in batch.items()}
    #             outputs = self.model(**batch)
    #             loss = outputs.loss
    #             loss.backward()

    #             self.optimizer.step()
    #             self.lr_scheduler.step()
    #             self.optimizer.zero_grad()

    #         self.evaluate() # evaluate the accuracy of the model for every epoch

    # def evaluate(self):
    #     model_name = "round" + str(self.round) + "_model"
    #     metric = load_metric('accuracy')
    #     self.model.eval()
    #     # start evaluation of the model using eval data
    #     for batch in self.eval_dataloader:
    #         batch = {k: v.to(self.device) for k, v in batch.items()}
    #         with torch.no_grad():
    #             outputs = self.model(**batch)

            # logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
    #         # print(logits)
    #         # print(predictions)
    #         # print(batch['labels'])
            # references = torch.reshape(batch['labels'], (-1,))
    #         # print(references)
            # metric.add_batch(predictions=predictions, references=references)
    #     score = metric.compute()
    #     accuracy = score['accuracy']
    #     # if current epoch's accuracy is higher than the maxAccuracy recorded, replace maxAccuracy and the saved model file
    #     if (accuracy >= self.maxAccuracy):
    #         self.maxAccuracy = accuracy
    #         torch.save(self.model.state_dict(), model_name)
    #     return accuracy




class Seq2Seq(pl.LightningModule):
    """
    Seq2seq Model Object
    """
    #individual args so that they can be serialized in torchscript
    def __init__(self, lr, lr_schedule, hid_dim, n_layers, n_features, enc_dropout, dec_dropout, n_mode3_tokens, n_mode3_token_embedding, n_mode3_token_layers, n_callsign_tokens, n_callsign_token_embedding, n_callsign_token_layers, n_class, teacher_forcing_ratio, class_weights, weight_by, labels_map, distributed):
        """
        Initialises the seq2seq model object.
        All hyparams are initialized in config.yaml
        :param lr: learning rate for trainer
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features in dataset used
        :param enc_dropout: dropout ratio for encoder
        :param dec_dropout: dropout ratio for decoder
        """
        super().__init__()
        self.learning_rate = lr
        self.lr_schedule = lr_schedule

        self.n_class = n_class
        self.attention = Attention(hid_dim)
        self.mode3_encoder = IdEncoder(n_mode3_tokens, n_mode3_token_embedding, hid_dim, n_mode3_token_layers)
        self.callsign_encoder = IdEncoder(n_callsign_tokens, n_callsign_token_embedding, hid_dim, n_callsign_token_layers)
        self.encoder = Encoder(hid_dim, n_layers, n_features, enc_dropout)
        self.decoder = Decoder(self.n_class, hid_dim, n_layers, dec_dropout, self.attention, self.mode3_encoder,self.callsign_encoder)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.class_weights_map = {}
        for key in class_weights.keys():

            self.class_weights_map[key] = {i:v for i,v in enumerate(class_weights[key].numpy())}
        
        
        self.labels_map = labels_map
        self.reverse_labels_map = {v:k for k,v in self.labels_map.items()}
        # self.save_hyperparameters() #cant save dictionary has hprams...
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights[weight_by])
        self.distributed = distributed
        self.save_hyperparameters()
    
    def create_mask(self, y):

        """
        Create mask to filter out the paddings.
        :param label_index: contains the true class each data belongs to
        :return: boolean where padded values returns False (tensor) [batch_size,sequence_len]
        """
        mask = y != self.labels_map['pad'] # real values = True, paddings are mapped to False.
        return mask
    
    def forward(self, x, mask, mode3, callsign, seq_len):
        """
        Forward propagation.
        :param x: features of dataset (tensor) [batch_size,sequence_len,feature_dim]
        :param y: ground truth of dataset (tensor) [batch_size,sequence_len]
        :param src_len: actual length of each data sequence [batch_size]
        :param callsign: contains the callsign  [batch,sequence_len,max_callsign_len_in_batch_over_time]

        :return: output the class for every timestamps (tensor)

        Note that x is padded with 0, but it doesnt matter as we pack it.
        Encoder output is padded with 0, but it will be ignored by attention mechanism 
        Note that 0 in y represents padded, start token is 1, and 2 is unknown.
        """

        batch_size = x.shape[0]
        sequence_len = x.shape[1] #(len,batch,)
        # seq_len = seq_len.reshape((-1))

        decoder_output = torch.zeros(batch_size,sequence_len,self.n_class).cuda()
        encoder_output, hidden_cell = self.encoder(x,seq_len)
        
        decoder_input = torch.ones([batch_size],dtype=torch.long) * self.labels_map['start']  # initialize start token , [0,n_class-1] are actual classes , start token is n_class
        decoder_input = decoder_input.to('cuda') # this is created explcitly so need to send to device

        
        for t in range(sequence_len):
            mode3_input = mode3[:,t,:]
            callsign_input = callsign[:,t,:]
            
            # hidden_cell = torch.jit.annotate(Tuple[Tensor,Tensor],(torch.empty(1),torch.empty(1)))
            output, hidden_cell = self.decoder(decoder_input,mode3_input,callsign_input,hidden_cell,encoder_output,mask)
            #output is [batch,n_targets]
            decoder_output[:,t,:] = output

            
            # teacher_force = random.random() < self.teacher_forcing_ratio
            # top1 = output.argmax(1)
            # decoder_input = y[t] if teacher_force else top1
            
            decoder_input = output.argmax(1)
        return decoder_output

    def configure_optimizers(self):
        '''
        Optimizer
        Adam and Learning Rate Decay used. 
        '''

        if self.lr_schedule['lr_decay']['use_decay']:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode = 'min',
                factor = self.lr_schedule['lr_decay']['factor'],
                patience = self.lr_schedule['lr_decay']['patience'], # will drop lr AFTER the patience + 1 epoch
                cooldown = self.lr_schedule['lr_decay']['cooldown'],
                eps =  self.lr_schedule['lr_decay']['eps'],
                verbose = True
                )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.lr_schedule['lr_decay']['metric_to_track']
                }

        elif self.lr_schedule['lr_cyclic']['use_cyclic']:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
  
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer,
                base_lr=self.lr_schedule['lr_cyclic']['lower_lr'],
                max_lr=self.learning_rate,
                step_size_up=self.lr_schedule['lr_cyclic']['epoch_size_up'],#* self.batch_per_epoch,
                step_size_down=self.lr_schedule['lr_cyclic']['epoch_size_down'],#, * self.batch_per_epoch,
                mode=self.lr_schedule['lr_cyclic']['mode'],
                cycle_momentum=False, 
                ),
                'interval': 'epoch',
            }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                }
        #use scale fn and scale mode to overwrite mode.
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    def weighted_loss_function(self,output, y):        
        """
        Calculate the weighted loss (point by point evaluation)
        :param trg: ground truth given by dataset (tensor)
        :param output: output from the model (tensor)
        :label_index: contains the true class each data belongs to;
                    used for calculating proportion hence weights (tensor)
        :param mask: used to mask out the padding (tensor)
        :return: loss needed for backpropagation and logging (float)
        """
        # y [batch_size,seq_len]
        # output [batch_size,seq_len,cats]

    
        mask = self.create_mask(y)
        y = torch.masked_select(y, mask) #[batch_size*time]
        mask = mask.unsqueeze(2).expand(output.size())
        output = torch.masked_select(output, mask)
        output = output.view(-1,self.n_class).cuda() #change to [batch_size*time,n_classes]

        return self.loss(output,y)

    def calculate_accuracy(self,output, y):
        """
        Calculate the accuracy (point by point evaluation)
        :param output: output from the model (tensor)
        :param Y: ground truth given by dataset (tensor)
        :param mask: used to mask out the padding (tensor)
        :return: accuracy used for validation logs (float)
        """

        output,y = self._remove_paddding(output,y)
        confidence, max_indices = torch.max(output,1)
        acc = (max_indices == y).sum().item()/max_indices.shape[0]



        weights = np.vectorize(self.class_weights_map['label_point_count'].get)(y.cpu().numpy()) #or y.tolist() or y.flatten.tolist()
        weighted = weights * (max_indices == y).cpu().numpy()
        weighted_acc = np.sum(weighted)/np.sum(weights)

        return acc, weighted_acc, confidence
    
    def _remove_paddding(self,output,y):
        mask = self.create_mask(y)
        
        y = torch.masked_select(y, mask)
        mask = mask.unsqueeze(2).expand(output.size())
        output = torch.masked_select(output, mask)

        output = output.view(-1,self.n_class).cuda()
        return output,y

    def get_segment_level_results(self, output, y, seq_len):
        output,y = self._remove_paddding(output,y) #change to [batch_size*time,n_classes]
        _, max_indices = torch.max(output,1)

        # print('seq',np.sum(seq_len))
        # print('o',output.shape)
        # print('y',y.shape)

        seg_y = []
        seg_pred = []
        start_index = 0
        for current_seg_len in seq_len: #iterate over each track sample
            end_index = start_index + current_seg_len

            track_y = y[start_index:end_index].cpu().numpy()
            track_pred = max_indices[start_index:end_index].cpu().numpy()
            df = pd.DataFrame.from_dict({'y':track_y,'pred':track_pred})
            df = df.groupby('y')['pred'].apply(lambda x : x.value_counts().nlargest(1))\
            .reset_index().drop('pred',1)\
            .rename(columns={'level_1':'pred'})
            
            seg_y.append(df['y'].to_numpy())
            seg_pred.append(df['pred'].to_numpy())
            start_index = end_index
        seg_y = np.concatenate(seg_y)
        seg_pred = np.concatenate(seg_pred)
        seg_acc = (seg_y == seg_pred).mean()

        weights = np.vectorize(self.class_weights_map['label_segment_count'].get)(seg_y) #or y.tolist() or y.flatten.tolist()

        weighted = weights * (seg_y == seg_pred)
        weighted_acc = np.sum(weighted)/np.sum(weights)

        return seg_acc, weighted_acc, seg_y, seg_pred

    def training_step(self, batch, batch_idx):
        '''
        Pytorch Lightning Trainer (training)
        '''
        x = batch[0]
        y = batch[1]
        mode3 = batch[2]
        callsign = batch[3]
        seq_len = batch[-1]
        mask = self.create_mask(y)


        #this is calling the forward implicitly
        output = self(x,mask,mode3,callsign,seq_len)
        loss = self.weighted_loss_function(output,y)

        acc,weighted_acc,confidence = self.calculate_accuracy(output,y)

        self.log('train_loss',loss,sync_dist=self.distributed)
        self.log('train_acc',acc,sync_dist=self.distributed)
        self.log('train_weighted_acc',weighted_acc,sync_dist=self.distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Pytorch Lightning Trainer (validation)
        '''
        x = batch[0]
        y = batch[1]
        mode3 = batch[2]
        callsign = batch[3]
        seq_len = batch[-1]
        mask = self.create_mask(y)

        output = self(x,mask,mode3,callsign,seq_len)

        loss = self.weighted_loss_function(output,y)
        acc, weighted_acc, confidence = self.calculate_accuracy(output,y)
        seg_acc, seg_weighted_acc, seg_y, seg_pred = self.get_segment_level_results(output,y,seq_len)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=self.distributed)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=self.distributed)
        self.log('val_weighted_acc', weighted_acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=self.distributed)
        self.log('val_seg_acc', seg_acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=self.distributed)
        self.log('val_weighted_seg_acc', seg_acc, prog_bar=True, on_step=False, on_epoch=True,sync_dist=self.distributed)

        output,y = self._remove_paddding(output,y)
        _, max_indices = torch.max(output,1)

        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_weighted_acc': acc,
            'val_seg_acc': seg_acc,
            'labels':y,
            'predictions':max_indices,
            'confidence':confidence,
            'seg_labels': seg_y,
            'seg_predictions': seg_pred}

    def validation_epoch_end(self, validation_step_outputs):
        # Log confusion matrices into tensorboard
        
        preds_list = list(map(lambda x: x['predictions'], validation_step_outputs))
        labels_list = list(map(lambda x: x['labels'], validation_step_outputs))

        seg_preds_list = list(map(lambda x: x['seg_predictions'], validation_step_outputs))
        seg_labels_list = list(map(lambda x: x['seg_labels'], validation_step_outputs))

        y_pred = torch.cat(preds_list).cpu().numpy()
        y_true = torch.cat(labels_list).cpu().numpy()

        seg_y_pred = torch.cat(preds_list).cpu().numpy()
        seg_y_true = torch.cat(labels_list).cpu().numpy()


        cm_fig = results.plot_confusion_matrix(y_true, y_pred, [self.reverse_labels_map[i] for i in range(self.n_class)])

        seg_cm_fig = results.plot_confusion_matrix(seg_y_true, seg_y_pred, [self.reverse_labels_map[i] for i in range(self.n_class)])

        self.logger.experiment.add_figure('True vs Predicted Labels', cm_fig, global_step=self.current_epoch)
        self.logger.experiment.add_figure('Segment Level True vs Predicted Labels', seg_cm_fig, global_step=self.current_epoch)



        # if self.target_classes:
        #     cm_fig = plot_confusion_matrix(y_true, y_pred, self.class_names, self.target_classes)
        #     self.logger.experiment.add_figure('True vs Predicted Labels for Targeted Classes', cm_fig, global_step=self.current_epoch)  


