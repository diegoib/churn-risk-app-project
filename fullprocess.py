import os
import json
import ast

from training import train_model
from scoring import score_model
from deployment import store_model_into_pickle
from app import app
from reporting import reporting
from ingestion import merge_multiple_dataframe

with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
input_folder_path = config['input_folder_path']
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

##################Check and read new data
#first, read ingestedfiles.txt

with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'r') as file:
    ingestedfiles = file.read().split(',')

completelist = os.listdir(input_folder_path)

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
newfiles = set(ingestedfiles).difference(set(completelist))
check_newdata = len(newfiles) > 0

if check_newdata:
    merge_multiple_dataframe(input_folder_path, output_folder_path)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if check_newdata:
    with open(os.path.join(prod_deployment_path,'latestscore.txt'), 'r') as file:
        f1score = ast.literal_eval(file.read())

    new_f1score = score_model(output_folder_path, prod_deployment_path)
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    check_modelscore = new_f1score < f1score


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if check_newdata and check_modelscore:
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    train_model(output_folder_path, model_path)
    store_model_into_pickle(model_path, output_folder_path, prod_deployment_path)
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
    reporting(output_folder_path, 'finaldata.csv', model_path)
    app.run(host='0.0.0.0', port=8000)





