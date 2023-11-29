from src.DimondPricePrediction.constants import *
import yaml
from src.DimondPricePrediction.Components.DataIngestion import DataIngestionworkflow
import os
import zipfile
import urllib.request as request
path=CONFIG_FILE_PATH
print(path)
with open(path,'r') as f:
    contents=yaml.safe_load(f)
data_source_url=contents['data_source_url']
print(data_source_url)
filename, headers = request.urlretrieve(
                url = data_source_url,
                filename = contents['raw_data'])
unzip_path=contents['unzip']
os.makedirs(unzip_path, exist_ok=True)
with zipfile.ZipFile(contents['raw_data'], 'r') as zip_ref:
    path_name=zip_ref.extractall(unzip_path)
print(path_name)