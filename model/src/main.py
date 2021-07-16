import argparse

from clearml import Task
from clearml.utilities import seed

from model import experiment
from model.config import cfg
from pipeline import pipeline
from pipeline.config import cfg as pipeline_cfg
from remote import s3utility
from remote.config import cfg as remote_cfg

import random


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser = experiment.Experiment.add_experiment_args(parser)
    parser = pipeline.PMCDataPipeline.add_pipeline_args(parser)
    args = parser.parse_args()

    # task = Task.init(project_name="LM Project", task_name="Fine tuning",output_uri="http://192.168.56.253:9000/minio/vsmodels/snapshots")
    # task = Task.init(project_name="BERT", task_name="Fine tuning for domain specificity - with sampler")
    # task = Task.init(project_name="BERT", task_name="test Fine tuning for QA - using pretraining ckpt (cased) early stopping, patience=3")
    task = Task.init(project_name="BERT", task_name="testing qa")
    seed.make_deterministic(seed=random.randint(0, 10000))
    model_config_dict = task.connect_configuration(cfg,name='Model Training Parameters')
    pipeline_config_dict = task.connect_configuration(pipeline_cfg,name='Data Pipeline Parameters')
        
    # PMC_data_pipe = pipeline.PMCDataPipeline(args)
    # PMC_data_pipe()

    exp = experiment.Experiment(args)
    # exp.run_experiment()
    # exp.create_torchscript_model('k=0-epoch=0.ckpt')
    # print("in main.py")
    # exp = experiment.Experiment(args)
    # # exp = experiment.Experiment(args, task)
    pretrain_best = exp.run_experiment(task='PRETRAIN', model_startpt=None)
    print(pretrain_best)
    # # exp.run_experiment(task='QA', model_startpt=None)
    # exp.run_experiment(task='QA', model_startpt="trained_models/PRETRAIN-epoch=0.ckpt")




