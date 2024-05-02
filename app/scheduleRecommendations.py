from bson import ObjectId
from datetime import datetime, timedelta

def format_times(day, start_time, end_time):
    """ Converts a weekday and time intervals into actual datetime stamps """
    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    now = datetime.utcnow()
    # Calculate the next occurrence of the given day
    next_day = now + timedelta((weekdays[day] - now.weekday()) % 7)
    next_start = datetime.combine(next_day, datetime.strptime(start_time, '%H:%M').time())
    next_end = datetime.combine(next_day, datetime.strptime(end_time, '%H:%M').time())

    return next_start.isoformat(), next_end.isoformat()



def jsonify_data(data):
    """ Recursively convert MongoDB ObjectId to strings in the given data. """
    if isinstance(data, list):
        return [jsonify_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: jsonify_data(value) for key, value in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    return data

def time_overlap(interval1, interval2):
    """ Check if two time intervals overlap. """
    start1 = datetime.strptime(interval1['start'], '%H:%M')
    end1 = datetime.strptime(interval1['end'], '%H:%M')
    start2 = datetime.strptime(interval2['start'], '%H:%M')
    end2 = datetime.strptime(interval2['end'], '%H:%M')
    return max(start1, start2) < min(end1, end2)
