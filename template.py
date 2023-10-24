import os
from pathlib import Path

packagename="DiamondPricePrediction"

list_files=[
    "github/workflow/.gitkeep",
    f"src/{packagename}/Components/__init__.py",
    f"src/{packagename}/Components/DataIngestion.py",
    f"src/{packagename}/Components/FeatureEngineering.py",
    f"src/{packagename}/Components/ModelTraining.py",
    f"src/{packagename}/Pipeline/__init__.py",
    f"src/{packagename}/Pipeline/Training.py",
    f"src/{packagename}/Pipeline/Prediction.py",
    f"src/{packagename}/logger.py",
    f"src/{packagename}/Exception.py",
    f"src/{packagename}/utils/__init__.py",
    "requirements.txt",
    "Setup.py",
    "init_setup.sh",
    "notebooks/research.ipynb",
    "notebooks/data/.gitkeep"

]


for filepath in list_files:
    file=Path(filepath)
    folder,files=os.path.split(file)



    if folder!='':
        os.makedirs(folder,exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
    else:
        print(f"{filepath} already exists")