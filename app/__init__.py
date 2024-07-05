from flask import request, jsonify, Flask
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import time
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time.time()
app = Flask(__name__)
CORS(app)

# Load pre-trained models
recommendation_model = pickle.load(open("best_model_recommendation.sav", "rb"))
vectorizer = pickle.load(open("vectorizerr.sav", "rb"))
df_reference = pd.read_excel('Classeur2.xlsx')

####### Churn Rate #######
# model = pickle.load(open("modelChurnRF.sav", "rb"))

# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = request.json
#     preprocessed_input = preprocess_input(input_data)
    
#     prediction = model.predict(preprocessed_input)
#     probability = model.predict_proba(preprocessed_input)[:, 1]
    
#     response = {
#         "prediction": int(prediction[0]),
#         "probability": float(probability[0])
#     }
    
#     return jsonify(response)

@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json
    preprocessed_input = preprocess_recommendation_input(input_data)
    
    # Predict similarity scores
    similarity_scores = recommendation_model.predict(preprocessed_input)
    
    # Add similarity scores to the reference DataFrame
    df_reference['Predicted_Similarity_Score'] = similarity_scores
    
    # Sort by similarity scores and get top recommendations
    recommendations_sorted = df_reference.sort_values(by='Predicted_Similarity_Score', ascending=False).head()
    
    response = recommendations_sorted.to_dict(orient='records')
    
    return jsonify(response)

######## swagger addons 
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

def preprocess_recommendation_input(data):
    input_df = pd.DataFrame(data, index=[0])
    new_text = input_df['Functional_Requirements'] + ", " + input_df['Technologies_Used']
    sparse_matrix_new = vectorizer.transform(new_text)
    return sparse_matrix_new
