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
from dotenv import load_dotenv
from flask_pymongo import MongoClient
import os
from pymongo import MongoClient
from .scheduleRecommendations import jsonify_data, time_overlap ,format_times
load_dotenv() 
t0 = time.time()
app = Flask(__name__)
CORS(app)




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


########## Configure MongoDB #########
client = MongoClient(os.getenv('MONGODB_URL'))
db = client['elkindy']


@app.route('/')
def suggestions():
    teachers = list(db.users.find({"role": "teacher"}))
    students = list(db.users.find({"role": "student"}))
    courses = list(db.courses.find())
    suggestions = []
    teacher_busy_times = {}
    student_course_types = {}

    for student in students:
        student_course_types[student['email']] = {}
    for teacher in teachers:
        teacher_busy_times[teacher['email']] = []
    for course in courses:
        interested_students = [s for s in students if course['_id'] in s.get('studentManagement', {}).get('courses', [])]
        available_teachers = [t for t in teachers if course['_id'] in t.get('teacherManagement', {}).get('coursesPreferences', [])]

        for teacher in available_teachers:
            teacher_slots = teacher['teacherManagement']['availableTimeSlots']
            for t_slot in teacher_slots:
                t_intervals = t_slot['intervals']
                for t_int in t_intervals:
                    possible_students = []
                    for student in interested_students:
                        student_slots = student['studentManagement']['availableTimeSlots']
                        for s_slot in student_slots:
                            if t_slot['day'] == s_slot['day']:
                                s_intervals = s_slot['intervals']
                                for s_int in s_intervals:
                                    if time_overlap(t_int, s_int):
                                        possible_students.append(student['_id'])

                    if possible_students:
                        start_time, end_time = format_times(t_slot['day'], t_int['start'], t_int['end'])
                        suggestion = {
                            'display': f"{course['name']} with {teacher['firstName']} {teacher['lastName']} on {t_slot['day']} at {t_int['start']} to {t_int['end']}",
                            'body': {
                                'scheduleSlot': {
                                    "start": start_time,
                                    "end": end_time,
                                    "Type": course['type'],
                                    "Course": str(course['_id']),
                                    "description": "",
                                    "allDay": True,
                                    "url": "",
                                    "teacher": str(teacher['_id']),
                                    "students": possible_students,
                                },
                                "message": "Schedule slot created successfully"
                            }
                        }
                        suggestions.append(suggestion)
    return jsonify(jsonify_data(suggestions))



######## swagger addons 
SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
