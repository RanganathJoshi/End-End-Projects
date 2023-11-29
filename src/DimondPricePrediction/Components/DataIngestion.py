import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.constants import *
import os
import sys
import yaml
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.DimondPricePrediction.utils.utils import read_yaml,download_data,unzip_data
 

class DataIngestionConfig:
    config_path=CONFIG_FILE_PATH
    with open(config_path,'r') as f:
        contents=yaml.safe_load(f)
    os.makedirs(contents['unzip'],exist_ok=True)
    source_url=contents['data_source_url']
    raw_data_path=contents['raw_data']
    unzip_data_path=contents['unzip']
    train_data_path=contents['train_data']
    test_data_path=contents['test_data']





class DataIngestionworkflow:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        
        try:
            download_data(self.ingestion_config.source_url,self.ingestion_config.raw_data_path)
            unzip_data(self.ingestion_config.raw_data_path,self.ingestion_config.unzip_data_path)
            data=pd.read_csv(Path("artifacts\gemstone.csv"))
            logging.info(" i have read dataset as a df")
            
            
            #os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            #data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Saved the raw data in artifacts folder')
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)


            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise customexception(e,sys)