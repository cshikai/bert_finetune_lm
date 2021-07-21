# Langauge Model Fine Tuning
Fine tuning a pretrained BERT model on domain-specific data. 

Using BertForPreTraining, Next Sentence Prediction & Masked Language Modelling are done simultaneously to fine-tune the model. Then, using BertForQuestionAnswering, it is fine-tuned for the QuestionAnswering downstream task. 

The domain used here is COVID-19, but it can be changed to any other domain language to fine-tune the model, as long as the dataset has the same format. 

## Table of Contents
1) [Requirements](#requirements)
2) [Usage](#usage)
3) [Using other datasets](#using-other-datasets)
4) [Using other BERT models](#using-other-bert-models)
5) [Documentation](#documentation)


## Requirements
- NVIDIA Driver
- CUDA: [Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
- Docker: [Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)
- ClearML: [Installation](https://allegro.ai/clearml/docs/docs/deploying_clearml/clearml_agent_install_configure.html)

## Usage

### Once everything has been installed successfully, you can either:

- Fine-tune on COVID-19 data, and run QA:
   - Put data_cleaned.json & COVID-QA.json into **model/src/pipeline folder**
   

- Fine-tune on other domain-specific data, and run QA:
   - Look at [using other datasets section](#using-other-datasets) to prepare data
   - Put COVID-QA.json into **model/src/pipeline folder**

- Fine-tune only, and not run QA:
  - Put data_cleaned.json into **model/src/pipeline folder**
  - In **main.py**, comment out line to run QA
   ```python
    exp = experiment.Experiment(args, task)
    pretrain_best = exp.run_experiment(task='PRETRAIN', model_startpt=None)
    # exp.run_experiment(task='QA', model_startpt=pretrain_best)
   ```
  - In **model/src/pipeline** folder, **pipeline.py**, comment out cleaning of QA
   ``` python
    def __call__(self):
        self.pretrain_clean()
        # self.qa_clean()
   ```

- Fine-tune with other models:
  - Refer to [using other BERT models](#using-other-bert-models)


### Build the docker container to run the project:
   ```bash
   docker-compose up --build
   ```
Console will show a ClearML link, go to that link to see the progress and results of the training

Trained models will be saved and logged in **src/model/trained_models**
   


## Using other datasets

### Dataset format (after pipeline)
Dataset format:
{"train": [[[]]], "valid": [[[]]], "test": [[[]]]}

In each set, i.e train, valid, test, it contains a list of articles, with a list of sections, with a list of sentences.

There should be a cased and uncased version of the dataset. 


### To use other datasets:
- If your data is already in the required format
   - Put both the cased.json and uncased.json into **model/src/pipeline folder**
   - Comment out code calling for pipeline in **main.py**
     ```python
      # PMC_data_pipe = pipeline.PMCDataPipeline(args)
      # PMC_data_pipe()

      ...

      exp = experiment.Experiment(args, task)
      pretrain_best = exp.run_experiment(task='PRETRAIN', model_startpt=None)

     ```

- If your data is not in the required format
  - Edit **pipeline.py** code and add a method to format your data, and comment out cleaning for pretrain.
    ```python
    def __call__(self):
        # self.pretrain_clean()
        self.your_method()
        self.qa_clean()

    def your_method(self):
       ... code to format data ...
    ```
  - Do NOT comment out pipeline in **main.py**

- Allocate suitable partition size based on size of your data in the **model/src/model** folder. The ratio of partition sizes for train/valid/test should be 7/2/1. In **transforms.py**:
```python
  class PretrainTransforms():
    def __init__(self, data: list, use_uncased: bool, mode: str):
      self.data = data # list of lists of sections where each section is a list of sentences
      self.use_uncased = use_uncased
      self.mode = mode.lower()

      # set number of partitions based on data size
      if (self.mode == "train"):
        self.nparts = 7
      elif (self.mode == "valid"):
        self.nparts = 2
      elif (self.mode == "test"):
        self.nparts = 1
```

## Using other BERT models
- In **model/src/model/transforms.py**, add a new class to transform the data into the format needed for tokenization for the model. In the Transformations class, add another elif statement to call the class.

```python
class YourModelTransforms():
  def __init__():
    . . .

  def __call__():
    . . .
    
    # save the transformed data as a dataframe into model folder as a parquet file

    df = pd.DataFrame({'sentence_a':sentence_a, 'sentence_b':sentence_b, 'labels':labels})
    df['idx'] = range(len(df))
    df = df.set_index('idx')

    path1 = 'model/'+self.mode+'_temp_uncased.parquet' if self.use_uncased else 'model/'+self.mode+'_temp_cased.parquet'
    path2 = 'model/'+self.mode+'_data_transformed_uncased.parquet' if self.use_uncased else 'model/'+self.mode+'_data_transformed_cased.parquet'

    # save original parquet
    df.to_parquet(path1, engine='fastparquet')
    # load original parquet
    ddf = dd.read_parquet(path1, columns=['sentence_a', 'sentence_b', 'labels'], engine='fastparquet')

    # repartition parquet and save
    ddf = ddf.repartition(npartitions=self.nparts).to_parquet(path2)

    # delete original parquet
    os.remove(path1)

    return length of data
    
    . . .
```

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

- In **model/src/model/dataset.py**, add a new method to tokenize and format the data for model inputs. Example:
```python
def tokenize_yourmodel(self, idx):
  data_tokenized = self.tokenizer(text_to_tokenize, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
  . . .
  return data_tokenized
```

- In **tokenize_steps** method, add another elif statement to call the tokenization method created earlier
```python
def tokenize_steps(self, idx):
  if self.task == "PRETRAIN":
    data_tokenized = self.tokenize_pretrain(idx)
  elif self.task == "QA": 
    data_tokenized = self.tokenize_qa(idx)
  elif self.task == "YOURMODEL":
    data_tokenized = self.tokenize_yourmodel(idx)

  return data_tokenized

```


- Under **model/src/model/model.py**, in the **\__init__** method, add the model by adding another elif statement:
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

- In the **forward** method, when getting the output, ensure that the inputs to self.bert are the correct set. For example, for [BertForQuestionAnswering](https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering), the required inputs are input_ids, attention_mask, token_type_ids, start_positions, end_positions. For [BertForPretraining](https://huggingface.co/transformers/model_doc/bert.html#bertforpretraining), it is a different set of inputs. 

```python
def forward(self, input_ids, attention_mask, labels, next_sentence_label, start_positions, end_positions, token_type_ids):
  if (self.task == "QA"):
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
  else:
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=next_sentence_label)

  return output

```

- In the **training_step**, **validation_step** and **test_step** methods, ensure that the inputs when calling self(the forward method) have the correct values. For example, for QA, start_positions end_positions are assigned while for Pretrain, labels and next_sentence_labels are assigned. 

```python
def training_step(self, batch, batch_idx):
       
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']
  token_type_ids = batch['token_type_ids']
  labels = None
  next_sentence_label = None
  start_positions = None
  end_positions = None
  if (self.task == "QA"):
    start_positions = batch['start_positions']
    end_positions = batch['end_positions']
  elif self.task == "PRETRAIN":
    labels = batch['labels']
    next_sentence_label = batch["next_sentence_label"]
        

  # call forward
  output = self(input_ids, attention_mask, labels, next_sentence_label, start_positions, end_positions, token_type_ids)
. . .

```

- Create methods that calculate metrics of the new model and call them in the training, validation and test steps.


## Documentation
### model/src
- **main.py**
  - ClearML task is created here, pipeline and experiment are called here to run. 


### model/src/pipeline
- **pipeline.py**
  - Dataset is formatted and cleaned here. It is also split into train/valid/test sets. Data is then output as a json file in the pipeline folder.
- **config.yaml**
  - Using either cased or uncased data is set here. 

### model/src/model
- **transforms.py**
  - Data is transformed here before tokenization. Different models/tasks may require different transformations.
- **dataset.py**
  - Transforms is called here when dataset.py is initialized, then, tokenization is done on the fly in the getitem method.
- **model.py**
  - This is where the model is defined, along with the forward, training, validation and test steps.
- **experiment.py**
  - This is where the trainer is called to train and test the model.

### model/src/trained_models
- Model checkpoint files are generated here after training.

