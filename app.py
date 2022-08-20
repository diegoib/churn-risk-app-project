from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
#app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict(): 
    df_name = request.args.get('dfname')       
    df_path = request.args.get('dfpath')
    predictions = model_predictions(df_path, df_name)
    #add return value for prediction outputs
    return str(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    test_data_path = request.args.get('test_data_path')
    model_path = request.args.get('model_path')
    f1_score = score_model(test_data_path, model_path)
    #add return value (a single F1 score number)
    return str(f1_score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    summary = dataframe_summary()
    #return a list of all calculated summary statistics
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    nas = missing_data()
    timing = execution_time()
    packs = outdated_packages_list()
    #add return value for all diagnostics
    return str([nas, timing, packs])


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000)#, debug=True, threaded=True)
