## How to run the code

First install the dataset from :
    https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/data

Change the "BASE_DATASET_PATH" variable in data_process.py file according to where you downloaded the dataset file.
Then run the code:
```
data_process.py 
```
This should save the dataset into the /data folder.


Then run:
```
trocr_finetune.py 
```