import argparse

from clearml import Task

from model import experiment
from model.config import cfg
from pipeline import pipeline
from pipeline.config import cfg as pipeline_cfg
from remote import s3utility
from remote.config import cfg as remote_cfg



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser = experiment.Experiment.add_experiment_args(parser)
    parser = pipeline.PMCDataPipeline.add_pipeline_args(parser)
    args = parser.parse_args()

    # task = Task.init(project_name="LM Project", task_name="Fine tuning",output_uri="http://192.168.56.253:9000/minio/vsmodels/snapshots")
    # task = Task.init(project_name="BERT", task_name="Fine tuning for domain specificity - with sampler")
    task = Task.init(project_name="BERT", task_name="Fine tuning for QA - after NSP and MLM (cased)")
    model_config_dict = task.connect_configuration(cfg,name='Model Training Parameters')
    pipeline_config_dict = task.connect_configuration(pipeline_cfg,name='Data Pipeline Parameters')

	
	
    clearml_task = None
    #clearml_task.set_base_docker('https://index.docker.io/v1/intel_model_model')	
    #clearml_task.execute_remotely(queue_name="gpu")
    # if remote_cfg.s3.use_s3:
    #     print('using remote data source...')
    #     s3_utils = s3utility.S3Utils(remote_cfg.s3.bucket,remote_cfg.s3.s3_path)
    #     s3_utils.s3_download_folder('train','/src/data/train')
    #     s3_utils.s3_download_folder('valid','/src/data/valid')
    
    # annotated_data_pipe =  pipeline.AnnotatedDataPipeline(args.pipeline_data_profiles,pipeline_cfg.data.groups,pipeline_cfg.source.db_uri,args.pipeline_path_h5,args.pipeline_data_valid_size,args.pipeline_data_max_length,args.pipeline_data_min_length,args.pipeline_seed)
    # #run pipeline to extract data from raw data
    # annotated_data_pipe()

    PMC_data_pipe = pipeline.PMCDataPipeline(args)
    PMC_data_pipe()

    

    # exp = experiment.Experiment(args,task)
    # exp.run_experiment()
    # exp.create_torchscript_model('k=0-epoch=0.ckpt')
    # print("in main.py")
    # exp = experiment.Experiment(args, clearml_task)
    # exp = experiment.Experiment(args, task)
    # nspbest = exp.run_experiment(task='NSP', model_startpt=None)
    # mlmbest = exp.run_experiment(task='MLM', model_startpt=nspbest)
    # exp.run_experiment(task='QA', model_startpt=mlmbest)
    # exp.run_experiment(task='MLM', model_startpt = nspbest)



