import os
import sys
import pandas as pd
import numpy as np

import dataclasses as dataclass
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.utils.utils import save_object
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OridinalEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')


class DataTrasformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()



    def prepare_data_transform(self):
        try:
            logging.info("Starting Data Transformation")

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")

            #Numerical Pipeline
            num_pipeline=Pipeline(

            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ]
            )

            #Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OridinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                    ('scalar',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
        

        except Exception as e:
            logging.info("Exception occured while creating Feature Engineering Pipeline, please resolve it")
            raise customexception(e, sys)


    def apply_transformation(self,train_data,test_data):
        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)
            logging.info("Datasets loaded")
            logging.info("Showing samples of data")
            logging.info(f"Train Dataset : \n{train_df.sample(5).to_string()}")
            logging.info(f"Test Dataset  : \n{test_df.sample(5).to_string()}")

            preprocessing_obj=self.prepare_data_transform()

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            train_preprocess=preprocessing_obj.fit_transform(input_feature_train_df)
            test_preprocess=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[train_preprocess, np.array(target_feature_train_df)]
            test_arr = np.c_[test_preprocess, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            logging.info("Saved Preprocessing object")

            return (train_arr,test_arr)
        
        except Exception as e:
            logging.info("Exception occured while feature Transformation")
            raise customexception(e,sys)