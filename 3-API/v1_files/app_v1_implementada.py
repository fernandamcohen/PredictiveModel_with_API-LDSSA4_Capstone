import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError,BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Begin database stuff

# the connect func 0tion checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in predictions.db
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    outcome = BooleanField()
    prediction = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Verifications 

def check_request(request):
    """
        Validates that our request is well formatted
        
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    
    if "observation_id" not in request:
        error = "Field 'observation_id' missing from request: {}".format(request)
        return False, error
    
    return True, ""

def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    
    valid_columns = {
        "observation_id",
        "Type",
        "Date",
        "Part of a policing operation",
        "Latitude",
        "Longitude",
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "station"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""

def check_latitude_longitude(observation):
    """
        Validates that latitude and longitude have valid values
        
        Returns:
        - assertion value: True if latitude/longitude is valid, False otherwise
        - error message: empty if latitude/longitude is valid, False otherwise
    """
    
    lat = observation.get("Latitude")
    lon = observation.get("Longitude")
        
    
    if isinstance(lat, str):
        error = "Field 'Latitude' is not a number"
        return False, error
    
    if isinstance(lon, str):
        error = "Field 'Longitude' is not a number"
        return False, error

    return True, ""


def check_Part_policing_operation(observation):
    """
        Validates that Part of a policing operation has valid values
        
        Returns:
        - assertion value: True if Part of a policing operation is valid, False otherwise
        - error message: empty if Part of a policing operation is valid, False otherwise
    """
    
    part_oper = observation.get("Part of a policing operation")
    
    if isinstance(part_oper, str):
        error = "Field 'Part of a policing operation' is not a boolean"
        return False, error        
    
    return True, ""

def transform_date(observation):
    
    """
        Creates month, hour and day_of_week features from Date
        
        Returns:
        - values if feature Date can be read as a date
        - None for all features if Date cannot be read as a date
    """
    
    date_ = observation.get("Date")
    
    try:
        date = pd.Timestamp(date_)
        hour = date.hour
        month = date.month
        day_of_week = date.day_name()
    except:
        hour = 0
        month = 0
        day_of_week = ''    

    return hour, month, day_of_week


# End of Verifications
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    observation=obs_dict

    print(obs_dict)

    #verify if there is observation_id
    obs_ok, error = check_request(observation)
    if not obs_ok:
        response = {'error': error}
        return jsonify(response)

    #verify columns are ok
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    #verify numerical features
    latlon_ok, error = check_latitude_longitude(observation)
    if not latlon_ok:
        response = {'error': error}
        return jsonify(response)   

    #verify boolean features
    operation_ok, error = check_Part_policing_operation(observation)
    if not operation_ok:
        response = {'error': error}
        return jsonify(response)  

    #date features
    hour, month, day_of_week  = transform_date(observation)  
    print(day_of_week)

    #id
    _id = obs_dict['observation_id']

    print(_id)

    #dict in the format need to predict
    obs_dataframe = {
    "Type": observation.get("Type"),
    "Part of a policing operation": observation.get("Part of a policing operation"),
    "Age range": observation.get("Age range"),
    "Latitude": observation.get("Latitude"),
    "Longitude": observation.get("Longitude"),
    "Legislation": observation.get("Legislation"),
    "hour": hour,
    "month": month,
    "day_of_week": day_of_week,
    "Gender": observation.get("Gender"),
    "Officer-defined ethnicity": observation.get("Officer-defined ethnicity")}

    print(obs_dataframe)
    #cleaning the features
    categorical_features = ['Type', 'Age range', 'Legislation', 'Gender', 'Officer-defined ethnicity']

    for column in categorical_features:
            obs_dataframe[column] = str(obs_dataframe[column]).strip().lower()  


    #a single observation into a dataframe
    obs = pd.DataFrame([obs_dataframe], columns=columns).astype(dtypes)

    #last features adjustments

    obs['Part of a policing operation'] = obs['Part of a policing operation'].fillna(False)

    obs['Legislation']=obs['Legislation'].fillna('missing infomation')
    legislation_categories = ['misuse of drugs act 1971 (section 23)', 'police and criminal evidence act 1984 (section 1)', 'criminal justice and public order act 1994 (section 60)', 'firearms act 1968 (section 47)', 'missing infomation']

    mask=(~obs['Legislation'].isin(legislation_categories))
    obs.loc[mask, 'Legislation']='others'

    obs['Latitude'] = obs['Longitude'].fillna(0)
    obs['Longitude'] = obs['Longitude'].fillna(50)

    obs['Age range']=obs['Age range'].replace({'under 10': 'under 18', '10-17': 'under 18'})


    # # now get ourselves an actual prediction of the positive class
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]

    response = {    "outcome": bool(prediction)
                }

    p = Prediction(
        observation_id=_id,
        observation = request.data,
        outcome=bool(prediction),
        prediction = proba
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['outcome']
        p.save()

        response = {"observation_id": p.observation_id,
                    "outcome": obs['outcome'],
                    "predicted_outcome": p.outcome
                    }

        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
