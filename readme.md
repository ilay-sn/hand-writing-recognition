## How to run the code

First install the dataset from :
    https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database/data

Change the path variables according to your installation.

Then run the below code for fine-tuning on the IAM dataset:
```
python trocr_finetune.py 
```

When the model is saved, run the below code to do inference:
```
python app.py
```