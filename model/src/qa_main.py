import torch
from torch._C import device
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast, AdamW, get_scheduler
import torch.nn.functional as F

from model.config import cfg

class BertQAPrediction():
    def __init__(self):
        self.use_uncased = cfg['model']['use_uncased']
        self.seq_length = cfg['model']['sequence_length']
        self.model_path = "qa_model.ckpt" # put the model in the src folder where this qa_main.py file is and rename it to qa_model.ckpt
        self.bert_case_uncase = 'bert_cached/bert-base-uncased' if self.use_uncased else 'bert_cached/bert-base-cased'
        self.bert = BertForQuestionAnswering.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_path))
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_case_uncase)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def input_question(self):
        print("\nasking for question\n")
        question = input("Please enter your question: ")
        return question

    def format_question(self, question):
        question = " ".join(question)
        if (self.use_uncased):
            question = question.lower()
        else:
            question = question[0].upper() + question[1:].lower()
        if (question[-1] != "?"):
            question += "?"
        return question

    def tokenize_question(self, question):
        tokenized_question = self.tokenizer(question, return_tensors='pt', max_length=self.seq_length, truncation=True)
        return tokenized_question
        
    def get_question(self):
        question = self.input_question()
        tokenized = self.tokenize_question(self.format_question(question))
        return tokenized

    def get_pred_answer(self, input_ids, start_logits, end_logits):
        start_position = torch.argmax(start_logits[0])
        end_position = torch.argmax(end_logits[0][start_position:]) + start_position
        ids = input_ids[0][start_position:end_position+1]
        answer = self.tokenizer.decode(ids)
        return answer

    def get_prediction(self, tokenized_ques):
        input_ids = tokenized_ques['input_ids'].to(self.device)
        attention_mask = tokenized_ques['attention_mask'].to(self.device)
        start_positions = tokenized_ques['start_positions'].to(self.device)
        end_positions = tokenized_ques['end_positions'].to(self.device)

        self.bert.eval()
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        pred_answer = self.get_pred_answer(input_ids, output.start_logits, output.end_logits)

        return pred_answer
    
    def __call__(self):
        tokenized_ques = self.get_question()
        pred_answer = self.get_prediction(tokenized_ques)
        print("BERT's predicted answer:", pred_answer)

BertQAPrediction()