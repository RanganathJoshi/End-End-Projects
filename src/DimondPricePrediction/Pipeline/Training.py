from src.DimondPricePrediction.Components.DataIngestion import DataIngestion
import os
import pandas as pd
import numpy as np
from src.DimondPricePrediction.components.FeatureEngineering import DataTransformation
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.ModelTraining import ModelTrainer

obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

#To perform Data Transformation
data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.apply_transformation(train_data_path,test_data_path)
model=ModelTrainer()
model.initiate_model_training(train_arr,test_arr)