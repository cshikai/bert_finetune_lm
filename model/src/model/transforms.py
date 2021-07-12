from typing import Callable, Optional, Tuple, List
import torch
from torch import Tensor
import os
import numpy as np
import requests
from transformers import BertTokenizer, BertTokenizerFast
import itertools
import random
from datasets import Dataset
import json

class PretrainTransforms():
    def __init__(self, data: list, use_uncased):
        self.data = data # list of lists of sections where each section is a list of sentences
        self.use_uncased = use_uncased
    def __call__(self):
        # print("transforms.py: in NSPTokenization class")
        sentence_a = []
        sentence_b = []
        labels = []
        dictResult = {}
        listResult = []

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
                        dictSentence = {}
                        if (j <= section_size-2):
                            #sentence_a.append(sent)
                            dictSentence['sentence_a'] = sent
                            if (random.random() >= 0.5):
                                # sentence_b is the correct next sentence after sentence_a
                                # sentence_b.append(section[j+1])
                                # labels.append(0)

                                dictSentence['sentence_b'] = section[j+1]
                                dictSentence['labels'] = 0
                            else:
                                # sentence_b is the wrong next sentence after sentence_a
                                rand_sent = sent
                                while (rand_sent == sent):
                                    randi = random.randint(0, bag_size-1)
                                    rand_sent = bag[randi]
                                # sentence_b.append(rand_sent)
                                # labels.append(1)

                                dictSentence['sentence_b'] = rand_sent
                                dictSentence['labels'] = 1
                        else: # end of section reached
                            break

                        listResult.append(dictSentence)
        
        # return model_inputs 
        # dictResult['sentence_a'] = sentence_a
        # dictResult['sentence_b'] = sentence_b
        # dictResult['labels'] = labels

        # dictResult.map(json.dumps).to_textfiles("pipeline/test.json")
        path = 'model/data_transformed_uncased.txt' if self.use_uncased else 'model/data_transformed_cased.txt'

        with open(path, 'w') as output:
            for row in listResult:
                output.write(str(row)+'\n')

        return len(listResult)

class QATransforms():
    def __init__(self, data: list):
        self.data = data
    def __call__(self):
        contexts = []
        questions = []
        answers = []
        dictResult = {}
        for i, pair in enumerate(self.data):
            contexts.append(self.data[i]['context'])
            questions.append(self.data[i]['question'])
            answers.append(self.data[i]['answer'])

        dictResult['contexts'] = contexts
        dictResult['questions'] = questions
        dictResult['answers'] = answers

        return dictResult


class Transformations():
    def __init__(self, task: str, mode:str, use_uncased):
        self.task = task
        self.mode = mode 
        self.use_uncased = use_uncased
    def __call__(self):
        
        if self.task == "QA":
            # path for QA
            path = 'pipeline/uncased_qna.json' if self.use_uncased else 'pipeline/cased_qna.json'
        elif self.task == "PRETRAIN":
            # paths for nsp and mlm
            path = 'pipeline/uncased.json' if self.use_uncased else 'pipeline/cased.json'

        with open(path) as f:
            all_data = json.load(f)
            data_loaded = all_data[self.mode]


        

        if self.task == "PRETRAIN":
            pretrain = PretrainTransforms(data=data_loaded, use_uncased=self.use_uncased)
            transformations = pretrain()
        elif self.task == "QA":
            qa = QATransforms(data=data_loaded)
            transformations = qa()
        else:
            pass # if we decide to fine tune more tasks
            
        return transformations 
