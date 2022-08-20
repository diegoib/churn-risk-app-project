
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(test_data_path, df_name):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv(os.path.join(test_data_path, df_name))
    df = df[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    predictions = model.predict(df)
    #return value should be a list containing all predictions
    return predictions


##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    cols = [col for col in df.columns if df[col].dtype != "O"]
    summary = []
    for c in cols:
        summary.append(c)
        summary.append(df[c].mean())
        summary.append(df[c].median())
        summary.append(df[c].std())

    #return value should be a list containing all summary statistics
    return summary 

################# Missing data
def missing_data():
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    return df.isna().sum() / df.shape[0]    

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing_ingestion = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system('python training.py')
    timing_train = timeit.default_timer() - starttime

    #return a list of 2 timing values in seconds
    return [timing_ingestion, timing_train]


##################Function to check dependencies
def outdated_packages_list():
     return os.system('pip list --outdated')


if __name__ == '__main__':
    model_predictions(test_data_path, 'testdata.csv')
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
