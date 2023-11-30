import pandas as pd
import numpy as np
import os

import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from dataclasses import dataclass
from src.DimondPricePrediction.utils.utils import save_object
#from src.DimondPricePrediction.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

@dataclass
class ModelTrainerConfig:
    trained_config=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting the data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
        
            def evaluate_model(x_train,x_test,y_train,y_test,models):
                scores=dict()
                for i in range(len(models)):
                    current_model=list(models.values())[i]
                    current_model.fit(x_train,y_train)
                    y_test_pred=current_model.predict(x_test)
                    score=r2_score(y_test,y_test_pred)
                    scores[list(models.keys())[i]]=score


                return scores
                
            model_report:dict=evaluate_model(x_train,x_test,y_train,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f"model report : {model_report}")
            best_model_score=max(list(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
 
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f"Model Summary :{model_report}")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path=self.model_trainer_config.trained_config,
                        obj=best_model)
            
            return self.model_trainer_config.trained_config


        except Exception as e:
            logging.info("Error occured while training the model")
            raise customexception(e,sys)       
