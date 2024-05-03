from bson import ObjectId
from flask_pymongo import MongoClient
import os
from pymongo import MongoClient
from datetime import datetime, timedelta
# client = MongoClient(os.getenv('MONGODB_URL'))
mongo_url = "mongodb+srv://enigma:enigma@enigma-elkindy.r1oa8tj.mongodb.net/elkindy"
client = MongoClient(mongo_url)
db = client['elkindy']
existing_schedules = list(db.scheduleslots.find())

def format_times(day, start_time, end_time):
    """ Converts a weekday and time intervals into actual datetime stamps with UTC format,
        subtracting one hour from both the start and end times and formatting with milliseconds. """
    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    now = datetime.utcnow()
    # Calculate the next occurrence of the given day
    next_day = now + timedelta((weekdays[day] - now.weekday()) % 7)
    
    # Parse times and subtract one hour
    parsed_start = datetime.strptime(start_time, '%H:%M').time()
    parsed_end = datetime.strptime(end_time, '%H:%M').time()
    adjusted_start = datetime.combine(next_day, parsed_start) - timedelta(hours=1)
    adjusted_end = datetime.combine(next_day, parsed_end) - timedelta(hours=1)

    # Format date with milliseconds and 'Z' for UTC
    return adjusted_start.strftime('%Y-%m-%dT%H:%M:%S.000Z'), adjusted_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')

def jsonify_data(data):
    """ Recursively convert MongoDB ObjectId to strings in the given data. """
    if isinstance(data, list):
        return [jsonify_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonify_data(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

def time_overlap(start1, end1, start2, end2):
    return max(start1, start2) < min(end1, end2)
