# app/__init__.py
from flask import request, jsonify, Flask
from .ocr import paddle_scan
from .churnRate import preprocess_input
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from PIL import Image
import numpy as np
import time
import pickle


t0 = time.time()
app = Flask(__name__)
CORS(app)

####### Home #######

@app.route("/")
def home():
    return jsonify({
        "Message": "app up and running successfully"
    })

####### Churn Rate #######
model = pickle.load(open("modelChurnRF.sav", "rb"))
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


####### Convertion Image #######


@app.route('/convert', methods=['POST'])
def convert_image_to_json():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    print(request)
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'error': 'No file provided'}), 400
    image = Image.open(image_file)
    image_array = np.array(image.convert('RGB'))   
    receipt_texts = paddle_scan(image_array)
    set_texts = list(set(receipt_texts))
    print(50*"--","\ntext only:\n",set_texts)
    return jsonify(set_texts)


######## swagger addons 
SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Flask API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
