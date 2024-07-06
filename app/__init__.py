# __init__.py

from flask import request, jsonify, Flask
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import time
import pickle
import pandas as pd

t0 = time.time()
app = Flask(__name__)
CORS(app)

# Load the recommendation model
recommendation_model = pickle.load(open("best_model_recommendation.sav", "rb"))

# Load the saved TfidfVectorizer
with open("vectorizerr.sav", "rb") as f:
    vectorizer = pickle.load(f)

# Load your reference DataFrame (Classeur2) with specified encoding and handle bad lines
df_reference = pd.read_csv("Classeur_Sim.csv")  # or encoding='latin1'

# Preprocess input for recommendation
def preprocess_recommendation_input(data):
    input_df = pd.DataFrame(data, index=[0])
    new_text = input_df['Functional_Requirements'] + ", " + input_df['Technologies_Used']
    sparse_matrix_new = vectorizer.transform(new_text)
    return sparse_matrix_new, new_text[0]


@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json
    preprocessed_input, new_text = preprocess_recommendation_input(input_data)
    
    # Preprocess Functional_Requirements and Technologies_Used for df_reference
    df_reference['Combined_Text'] = df_reference['Functional_Requirements'] + ", " + df_reference['Technologies_Used']
    df_reference_sparse_matrix = vectorizer.transform(df_reference['Combined_Text'])
    
    # Calculate similarity scores
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(preprocessed_input, df_reference_sparse_matrix).flatten()
    
    # Reset df_reference index if needed
    df_reference.reset_index(drop=True, inplace=True)
    
    # Check lengths before assignment
    print("Length of df_reference:", len(df_reference))
    print("Length of similarity_scores:", len(similarity_scores))
    
    # Assign similarity_scores to df_reference
    df_reference['Predicted_Similarity_Score'] = similarity_scores
    
    # Sort by similarity scores and get top recommendations
    recommendations_sorted = df_reference.sort_values(by='Predicted_Similarity_Score', ascending=False).head()
    
    response = recommendations_sorted.to_dict(orient='records')
    
    return jsonify(response)
    input_data = request.json
    preprocessed_input = preprocess_recommendation_input(input_data)
    
    # Predict similarity scores
    similarity_scores = recommendation_model.predict(preprocessed_input)
    
    # Reset df_reference index if needed
    df_reference.reset_index(drop=True, inplace=True)
    
    # Check lengths before assignment
    print("Length of df_reference:", len(df_reference))
    print("Length of similarity_scores:", len(similarity_scores))
    
    # Assign similarity_scores to df_reference
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
