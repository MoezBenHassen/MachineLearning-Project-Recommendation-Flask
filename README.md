## Installing dependencies :
To install the dependencies listed in a `requirements.txt` file for your Python project, follow these steps:

**Install Dependencies**: Use `pip` (Python's package installer) to install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt --user
   ```
This command reads the `requirements.txt` file and installs all the Python packages specified in it along with their specified versions.


## Run project:
```bash
python run.py
```
## Easily check available APIs using Swagger : 
Navigate to [http://localhost:5000/swagger](http://localhost:5000/swagger) in your web browser to access the Swagger UI, assuming your Flask app is running on localhost and port 5000.

## Short explanation for the code to generate a project recommendation

```python
from flask import request, jsonify, Flask
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import time
import pickle
import pandas as pd
```
- **Importing libraries**: Here, necessary libraries for the Flask app, CORS handling, Swagger UI, time tracking, loading pickled models, and data manipulation are imported.

```python
t0 = time.time()
app = Flask(__name__)
CORS(app)
```
- **Initializing Flask and CORS**: A Flask app instance is created, and CORS (Cross-Origin Resource Sharing) is enabled to allow requests from different origins.

```python
# Load the recommendation model
recommendation_model = pickle.load(open("best_model_recommendation.sav", "rb"))

# Load the saved TfidfVectorizer
with open("vectorizerr.sav", "rb") as f:
    vectorizer = pickle.load(f)
```
- **Loading models**: The recommendation model and TfidfVectorizer, which were previously saved, are loaded into memory.

```python
# Load your reference DataFrame (Classeur2) with specified encoding and handle bad lines
df_reference = pd.read_csv("Classeur_Sim.csv")
```
- **Loading the reference DataFrame**: The CSV file containing the reference data is loaded into a pandas DataFrame.

```python
# Preprocess input for recommendation
def preprocess_recommendation_input(data):
    input_df = pd.DataFrame(data, index=[0])
    new_text = input_df['Functional_Requirements'] + ", " + input_df['Technologies_Used']
    sparse_matrix_new = vectorizer.transform(new_text)
    return sparse_matrix_new, new_text[0]
```
- **Preprocessing input data**: This function converts the input JSON data into a DataFrame, combines 'Functional_Requirements' and 'Technologies_Used' into a single text string, and transforms it using the TfidfVectorizer.

```python
@app.route('/recommend', methods=['POST'])
def recommend():
    input_data = request.json
    preprocessed_input, new_text = preprocess_recommendation_input(input_data)
```
- **Defining the `/recommend` route**: This route handles POST requests for recommendations. It retrieves JSON data from the request and preprocesses it.

```python
    # Preprocess Functional_Requirements and Technologies_Used for df_reference
    df_reference['Combined_Text'] = df_reference['Functional_Requirements'] + ", " + df_reference['Technologies_Used']
    df_reference_sparse_matrix = vectorizer.transform(df_reference['Combined_Text'])
```
- **Preprocessing reference data**: The reference DataFrame also combines 'Functional_Requirements' and 'Technologies_Used' into a single text string and transforms it using the same TfidfVectorizer.

```python
    # Calculate similarity scores
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(preprocessed_input, df_reference_sparse_matrix).flatten()
```
- **Calculating similarity scores**: Cosine similarity between the preprocessed input data and the reference data is calculated. This gives a measure of how similar the input is to each reference item.

```python
    # Reset df_reference index if needed
    df_reference.reset_index(drop=True, inplace=True)
    
    # Check lengths before assignment
    print("Length of df_reference:", len(df_reference))
    print("Length of similarity_scores:", len(similarity_scores))
```
- **Resetting index and debugging**: The DataFrame index is reset to ensure consistency. The lengths of the DataFrame and similarity scores are printed for debugging.

```python
    # Assign similarity_scores to df_reference
    df_reference['Predicted_Similarity_Score'] = similarity_scores
```
- **Assigning similarity scores**: The calculated similarity scores are added as a new column in the reference DataFrame.

```python
    # Sort by similarity scores and get top recommendations
    recommendations_sorted = df_reference.sort_values(by='Predicted_Similarity_Score', ascending=False).head()
    
    response = recommendations_sorted.to_dict(orient='records')
    
    return jsonify(response)
```
- **Sorting and responding**: The DataFrame is sorted by similarity scores in descending order, and the top recommendations are selected. These recommendations are converted to a dictionary format and returned as a JSON response.

```python
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
```
- **Swagger UI integration**: Swagger UI is set up to provide an interactive interface for the API documentation. This allows users to test the API endpoints directly from a web interface.

Overall, this code sets up a Flask API for generating recommendations based on input data, processes this input and the reference data using a TfidfVectorizer, calculates similarity scores using cosine similarity, and returns the top recommendations. The Swagger UI integration provides a user-friendly interface for interacting with the API.
