import os
import sys
import pickle
import yaml
import urllib.request as request
import numpy as np
import zipfile
import pandas as pd
from pathlib import Path
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)
    

def read_yaml(path_to_yaml:Path):
    with open(path_to_yaml,'r') as f:
        contents=yaml.safe_loads(f)
        logging.info("Yaml file loaded")

        return contents
    
def download_data(source_file:str,destination):
    filename, headers = request.urlretrieve(
                url = source_file,
                filename = destination)
    
    return destination

def unzip_data(zip_file_path,destination):
    os.makedirs(destination, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    return destination


