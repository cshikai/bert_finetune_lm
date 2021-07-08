# Langauge Model Fine Tuning
Fine tuning a pretrained BERT model on domain specific data. Built to train on COVID-19 data. Using BertForPreTraining, combines Next Sentence Prediction & Masked Language Modelling to fine tune together. 

## Requirements
1) NVIDIA Driver
2) CUDA: [Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
3) Docker: [Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)
4) ClearML: [Installation](https://allegro.ai/clearml/docs/docs/deploying_clearml/clearml_agent_install_configure.html)

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

- Fine tune and run with other tasks:
  - Refer to Documentation to edit codes

---
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

## Documentation

### model/src
- **main.py**


### model/src/pipeline
- **pipeline.py**
- **config.yaml**

### model/src/model
- **transforms.py**
- **dataset.py**
- **model.py**
- **experiment.py**

### model/src/trained_models
- model files generated after training

