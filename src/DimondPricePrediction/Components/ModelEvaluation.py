import os
import yaml
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from urllib.parse import urlparse
from src.DimondPricePrediction.utils.utils import load_object
from mlflow.models import infer_signature


class SaveModel:
    def __init__(self,model_path):
        self.model_path=model_path

    def log_into_mlflow(self,mlflow_uri):
        
        mlflow.set_registry_uri(mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_registry_uri()).scheme
        model=load_object(self.model_path)
        preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        preprocessor_obj=load_object(preprocessor_path)
        test=pd.read_csv(Path('artifacts/test.csv'))
        x_test_raw=test.drop(columns=['price','id'])
        y_test_raw=test['price']
        x_test=preprocessor_obj.transform(x_test_raw)
        signatures=infer_signature(x_test,model.predict(x_test))
        with mlflow.start_run():
            if tracking_url_type_store!='file':

                mlflow.sklearn.log_model(model,"model",registered_model_name="Best Model_1 ")
                model_uri=mlflow.get_artifact_uri("model")
            else:
                mlflow.sklearn.log_model(model,"model") 


            result=mlflow.evaluate(model_uri,data=x_test,targets=np.array(y_test_raw),model_type='regressor',evaluators=["default"])