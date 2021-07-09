# Langauge Model Fine Tuning
Fine tuning a pretrained BERT model on domain specific data. 

Using BertForPreTraining, it combines Next Sentence Prediction & Masked Language Modelling to fine tune together. Then, using BertForQuestionAnswering, it is fine tuned for the QnA task. 

Domain can be changed to fine tune the model, as long as the dataset has the same format. 

## Table of Contents
1) [Requirements](#requirements)
2) [Usage](#usage)
3) [Using other datasets](#using-other-datasets)
4) [Documentation](#documentation)


## Requirements
- NVIDIA Driver
- CUDA: [Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
- Docker: [Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)
- ClearML: [Installation](https://allegro.ai/clearml/docs/docs/deploying_clearml/clearml_agent_install_configure.html)

## Usage

### Once everything has been installed successfully, you can either:

- Fine tune on COVID-19 data, and run QnA:
   - Put data_cleaned.json & COVID-QA.json into **model/src/pipeline folder**
   

- Fine tune on other data, and run QnA:
   - Look at [using other datasets section](#using-other-datasets) to prepare data
   - Put COVID-QA.json into **model/src/pipeline folder**

- Fine tune only, and not run QnA:
  - Put data_cleaned.json into **model/src/pipeline folder**
  - In **main.py**, comment out line to run QnA
   ```python
    exp = experiment.Experiment(args, task)
    pretrain_best = exp.run_experiment(task='PRETRAIN', model_startpt=None)
    # exp.run_experiment(task='QA', model_startpt=pretrain_best)
   ```
  - In model/src/pipeline folder, **pipeline.py**, comment out cleaning of QnA
   ``` python
    def __call__(self):
        self.pretrain_clean()
        #self.qna_clean()
   ```

- Fine tune with other models:
  - Refer to [using other BERT models](#using-other-bert-models) to 


### Build the docker container to run the project:
   ```bash
   docker-compose up --build
   ```
Console will show a ClearML link, go to that link to see results of training

Trained models will be logged in **src/model/trained_models**
   


## Using other datasets

### Dataset format
Dataset format:
{"train": [[[]]], "valid": [[[]]], "test": [[[]]]}

In each set, i.e train, valid, test, it contains a list of articles, with a list of sections, with a list of sentences.

There should be a cased and uncased version of the dataset. 


### To use other datasets:
- Your data already in the format
   - Put both the cased.json and uncased.json into **model/src/pipeline folder**
   - Comment out code calling for pipeline in **main.py**
     ```python
      # PMC_data_pipe = pipeline.PMCDataPipeline(args)
      # PMC_data_pipe()

      ...

      exp = experiment.Experiment(args, task)
      pretrain_best = exp.run_experiment(task='PRETRAIN', model_startpt=None)

     ```

- Your Data is not in format
  - Edit **pipeline.py** code and add a method to format your data.
    ```python
    def __call__(self):
        self.pretrain_clean()
        self.qna_clean()
        self.your_method()

    def your_method(self):
       ... code to format data ...
    ```
  - Do NOT comment out pipeline in **main.py**

## Using other BERT models
In **model/src/model/transforms.py**, add a new class to transform the data into the format needed for tokenization for the model. In the Transformations class, add another elif statement to call the class.

```python
def __call__(self):
  if self.task == "PRETRAIN":
    pretrain = PretrainTransforms(data=self.data)
    transformations = pretrain()
  elif self.task == "QA":
     qa = QATransforms(data=self.data)
     transformations = qa()
  elif self.task == "YOUR_MODEL":
    your_model = YourModelTransforms(data=self.data)
    transformations = your_model()
              
  return transformations 
```

In **model/src/model/dataset.py**, add a new method to tokenize and format the data for model inputs. Example:
```python
def tokenize_yourmodel(self, idx):
  context = self.data_transformed['contexts'][idx]
  data_tokenized = self.tokenizer(context, return_tensors='pt', truncation=False, padding=False)
  . . .
  return data_tokenized
```

In **__ len __** method, add another elif statement to get the length of the data, depending on your model type. 
```python
def __len__(self):
  if self.task == "PRETRAIN":
    return len(self.data_transformed['sentence_a'])
  elif self.task == "QA":
    return len(self.data_transformed['questions'])
  elif self.task == "YOURMODEL":
    return len(self.data_transformed['column_name'])
   
```

In **tokenize_steps** method, add another elif statement to call the tokenization method created earlier
```python
def tokenize_steps(self, idx):
  if self.task == "PRETRAIN":
    data_tokenized = self.tokenize_pretrain(idx)
  elif self.task == "QA": 
    data_tokenized = self.tokenize_qna(idx)
  elif self.task == "YOURMODEL":
    data_tokenized = self.tokenize_yourmodel(idx)

  return data_tokenized

```


Under **model/src/model/model.py**, in the __ init __ method, add the model by adding another elif statement:
```python
if (self.model_startpoint is None):
   if (self.task == "PRETRAIN"):
      self.bert = BertForPreTraining.from_pretrained(self.bert_case_uncase)
   elif (self.task == "QA"):
      self.bert = BertForQuestionAnswering.from_pretrained(self.bert_case_uncase)
      self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_case_uncase)
   elif (self.task == "YOUR_BERT_MODEL"):
      self.bert = BertFor......
# start training the model from previously trained model which was saved
else:
   if (self.task == "PRETRAIN"):
      self.bert = BertForPreTraining.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
   elif (self.task == "QA"):
      self.bert = BertForQuestionAnswering.from_pretrained(self.bert_case_uncase, state_dict=torch.load(self.model_startpoint))
      self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_case_uncase)
   elif (self.task == "YOUR_BERT_MODEL"):
      self.bert = BertFor......

```
TODO:
In the forward method, self.bert

In training_step, validation_step and test_step add elif to get correct tensors for model

Create methods that calculate metrics and call them in the steps


## Documentation
TODO
### model/src
- **main.py**
  - ClearML task is created here, pipeline and experiment is called here to run. 


### model/src/pipeline
- **pipeline.py**
  - Dataset is formatted and cleaned here. It is also split into train/test/valid sets.
  - __call__() 
  - pretrain_clean() cleans 
  - qna_clean()
- **config.yaml**
  - use_uncased

### model/src/model
- **transforms.py**
  - Transformations
  - PretrainTransforms
  - QATransforms
- **dataset.py**
  - init
  - getitem
  - len
  - tokenize_steps
  - tokenize_pretrain
  - tokenize_qna
- **model.py**
  - configure_optimizers
  - calculate_accuracy
  - calculate_perplexity
  - get_actual_answers
  - get_pred_answers
  - get_questions
  - calculate_f1
  - calculate_exactmatch
  - forward
  - training_step
  - validation_step
  - test_step
- **experiment.py**
  - run_experiment
  - CustomCheckpoint

### model/src/trained_models
- model files generated after training

