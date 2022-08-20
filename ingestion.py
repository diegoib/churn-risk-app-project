import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    #check for datasets, compile them together, and write to an output file
    csv_files = os.listdir(input_folder_path)
    final_df = pd.DataFrame()
    for f in csv_files:
        new_df = pd.read_csv(f)
        final_df = pd.concat([final_df, new_df], axis=0)
    final_df = final_df.drop_duplicates()
    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
   
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'wb') as file:
        file.write(csv_files)


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path)
