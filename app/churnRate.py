import pandas as pd
import numpy as np
from flask_cors import CORS


df_dummies = pd.read_csv('tel_churn.csv')


def preprocess_input(data):
    input_df = pd.DataFrame(data, index=[0])
    
    
    for column in input_df.columns:
        if input_df[column].dtype == bool:
            input_df[column] = input_df[column].astype(int)
            
    input_df_dummies = pd.get_dummies(input_df)
    
    input_df_dummies = input_df_dummies.reindex(columns=df_dummies.columns, fill_value=0)
    if 'Unnamed: 0' in input_df_dummies.columns:
       input_df_dummies.drop(columns=['Unnamed: 0'], inplace=True)
   
    return input_df_dummies
