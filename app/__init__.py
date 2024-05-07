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
from flask_pymongo import MongoClient
import os
from pymongo import MongoClient
from .scheduleRecommendations import jsonify_data, time_overlap ,format_times
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
# client = MongoClient(os.getenv('MONGODB_URL'))
mongo_url = "mongodb+srv://enigma:enigma@enigma-elkindy.r1oa8tj.mongodb.net/elkindy"
client = MongoClient(mongo_url)
db = client['elkindy']


@app.route('/')
def suggestions():
    # Default intervals if no specific time slots are given
    DEFAULT_INTERVALS = [
        {"start": "09:00", "end": "10:30"},
        {"start": "11:00", "end": "12:30"},
        {"start": "14:00", "end": "15:30"},
        {"start": "16:00", "end": "17:30"}
    ]
    teachers = list(db.users.find({"role": "teacher"}))
    students = list(db.users.find({"role": "student"}))
    courses = list(db.courses.find())
    suggestions = []

    def student_matches_level(student, teaching_levels):
        """Check if a student's level matches any of the levels the teacher can teach."""
        student_level = student.get('studentManagement', {}).get('level')
        return student_level in teaching_levels

    for course in courses:
        interested_students = [s for s in students if course['_id'] in s.get('studentManagement', {}).get('courses', [])]

        for teacher in teachers:
            # Ensure the teacher can teach the course and student matches the teaching level
            if str(course['_id']) not in [str(c) for c in teacher.get('teacherManagement', {}).get('coursesPreferences', [])]:
                continue

            teaching_levels = teacher['teacherManagement'].get('levels', [])
            available_students = [
                s for s in interested_students if student_matches_level(s, teaching_levels)
            ]

            teacher_slots = teacher['teacherManagement']['availableTimeSlots'] or [{'day': i, 'intervals': DEFAULT_INTERVALS} for i in range(7)]

            for t_slot in teacher_slots:
                t_intervals = t_slot['intervals'] if t_slot['intervals'] else DEFAULT_INTERVALS

                for t_int in t_intervals:
                    possible_students = []
                    for student in available_students:
                        student_slots = student['studentManagement']['availableTimeSlots'] or [{'day': t_slot['day'], 'intervals': DEFAULT_INTERVALS}]

                        for s_slot in student_slots:
                            if t_slot['day'] == s_slot['day']:
                                s_intervals = s_slot['intervals'] if s_slot['intervals'] else DEFAULT_INTERVALS

                                for s_int in s_intervals:
                                    start1, end1 = format_times(t_slot['day'], t_int['start'], t_int['end'])
                                    start2, end2 = format_times(s_slot['day'], s_int['start'], s_int['end'])

                                    if time_overlap(start1, end1, start2, end2):
                                        possible_students.append(student['_id'])

                    # Limit to one student only if the course type is "Instrument"
                    if course['type'] == "Instrument" and possible_students:
                        possible_students = [possible_students[0]]

                    if possible_students:
                        first_student = next(s for s in available_students if s['_id'] == possible_students[0])
                        level = first_student.get('studentManagement', {}).get('level')

                    # Create a list of student names that match the possible_students IDs
                    student_names_display = ', '.join(
                        [s['firstName'] + ' ' + s['lastName'] for s in available_students if s['_id'] in possible_students]
                    )
                    if possible_students:
                        start_time, end_time = format_times(t_slot['day'], t_int['start'], t_int['end'])
                        suggestion = {
                            'display': {
                                'course_name': course['name'],
                                'teacher_name': teacher['firstName'] + ' ' + teacher['lastName'],
                                'day': t_slot['day'],
                                'time_start': t_int['start'],
                                'time_end': t_int['end'],
                                'student_names': student_names_display,
                                'level': level

                            },
                            'scheduleSlot': {
                                "start": start_time,
                                "end": end_time,
                                "Type": course['type'],
                                "Course": str(course['_id']),
                                "description": "",
                                "allDay": False,
                                "url": "",
                                "teacher": str(teacher['_id']),
                                "students": possible_students
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
