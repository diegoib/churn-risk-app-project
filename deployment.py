from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import ast



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

model_path = os.path.join(config['output_model_path']) 


####################function for deployment
def store_model_into_pickle(model_path, dataset_csv_path, prod_deployment_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    # trained model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as file:
        model = pickle.load(file)
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'wb') as file:
        pickle.dump(model, file) 

    # model score
    with open(os.path.join(model_path,'latestscore.txt'), 'r') as file:
        f1score = ast.literal_eval(file.read())
    with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'w') as file:
        file.write(f1score)

    # records
    with open(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), 'r') as file:
        records = ast.literal_eval(f.read())
    with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt'), 'w') as file:
        file.write(records)
        
if __name__ == '__main__':
    store_model_into_pickle(model_path, dataset_csv_path, prod_deployment_path)
        

