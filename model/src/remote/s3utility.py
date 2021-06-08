#!/usr/bin/env python
# coding: utf-8


import os
from glob import glob
import boto3
from botocore.client import Config
from pytorch_lightning.callbacks import Callback
from .config import cfg
import time

class S3Callback(Callback):
    '''
    Callback used to upload data from docker container to s3 server after reaching model saving checkpoint
    '''

    def __init__(self):
        super().__init__()
        self.s3 = S3Utils(bucket=cfg.kube.bucket, s3_path=cfg.kube.s3_path)
    def on_epoch_end(self,trainer, pl_module):
        #upload model
        for file in os.listdir(cfg.train.checkpoint_dir):
            if os.path.isfile(os.path.join(cfg.train.checkpoint_dir,file)):
                s3_save_model_path = os.path.join('trained_models','model.cpkt')
                self.s3.s3_upload_file(os.path.join(cfg.train.checkpoint_dir,file),s3_save_model_path)
        #upload logs
        self.s3.s3_upload_folder(os.path.join(cfg.train.checkpoint_dir,'lightning_logs'))



class S3Utils:
    s3 = None
    
    MAX_ATTEMPT = 5 # 
    WAIT_TIME = 1



    def __init__(self, bucket, s3_path):
      
        self.s3 = boto3.resource('s3', \
            endpoint_url= cfg.s3.endpoint_url ,\
            aws_access_key_id= cfg.s3.aws_access_key_id ,\
            aws_secret_access_key= cfg.s3.aws_secret_access_key,\
            config=Config(signature_version= cfg.s3.signature_version),\
            region_name= cfg.s3.region_name)
        self.BUCKET = bucket
        self.S3_PATH = s3_path


    def s3_download_file(self,s3file,localfile):
        print("S3 Download s3://" + self.BUCKET + "/" + self.S3_PATH + localfile + " to " + localfile )
        print("Diagnostic info: \nBUCKET: ", self.BUCKET, "\nFROM: ", os.path.join(self.S3_PATH,s3file), "\nTO: ", localfile)

        attempt_no = 1
        success = False
        while attempt_no <= self.MAX_ATTEMPT and (not success):
            attempt_no = attempt_no + 1
            try:
                self.s3.Bucket(self.BUCKET).download_file(os.path.join(self.S3_PATH,s3file), localfile)
                success = True

            except:
                print('s3 connection failed. Retrying...')
                time.sleep(self.WAIT_TIME)
        if not success:
            print('Max attempt reached. Connection to MINIO not successful.')

        
    def s3_upload_file(self, localfile,s3file):
        print("S3 Uploading " + localfile + " to s3://" + os.path.join(self.S3_PATH,s3file))

        attempt_no = 1
        success = False
        while attempt_no <= self.MAX_ATTEMPT and (not success):
            attempt_no = attempt_no + 1
            try:
                self.s3.Bucket(self.BUCKET).upload_file(localfile, os.path.join(self.S3_PATH,s3file))
                success = True
            except:
                print('s3 connection failed. Retrying...')
                time.sleep(self.WAIT_TIME)
        if not success:
            print('Max attempt reached. Connection to MINIO not successful.')


        
    def s3_download_folder(self,s3_folder,local_dir):
        bucket = self.s3.Bucket(self.BUCKET)
        s3_dir = os.path.join(self.S3_PATH,s3_folder)
        for obj in bucket.objects.filter(Prefix=s3_dir):           
            target = os.path.join(local_dir, os.path.relpath(obj.key,s3_dir))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            print('dling')	
            self.s3_download_file(os.path.relpath(obj.key,self.S3_PATH),target)
    
    def s3_upload_folder(self, folder):
        folder_dir = os.path.split(folder)[0]  + '/'
        for dirr,_,files in os.walk(folder):
            for file in files:
                local_file = os.path.join(dirr,file)
                s3file = local_file.split(folder_dir)[-1]
                self.s3_upload_file(local_file,s3file) #bucket was previously 'training'
    

