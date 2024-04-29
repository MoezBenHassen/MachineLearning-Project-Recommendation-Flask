from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS


model = pickle.load(open("modelChurnRF.sav", "rb"))
df_dummies = pd.read_csv('tel_churn.csv')

app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    print(input_data)
    preprocessed_input = preprocess_input(input_data)
    
    prediction = model.predict(preprocessed_input)
    probability = model.predict_proba(preprocessed_input)[:, 1]
    
    response = {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }
    
    return jsonify(response)

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

if __name__ == '__main__':
    app.run(debug=True)
