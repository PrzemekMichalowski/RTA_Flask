# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%file apka.py

from flask import Flask
from flask_restful import Resource, Api
from flask import request, jsonify
import joblib
import numpy as np
from perceptron import Perceptron

app = Flask(__name__)
api = Api(app)

@app.route('/')
@app.route('/index')
def home():
    return "aplikacja ze srodowiskiem produkcyjnym API"

# TU jest instrukcja co wpisac do przegladarki
@app.route('/api/predict_perceptron', methods=['GET'])
def predict():
    # sepal length
    sepal_length = float(request.args.get('sl'))
    # sepal width
    sepal_width = float(request.args.get('sw'))
    # petal length
    petal_length = float(request.args.get('pl'))
    # petal width
    petal_width = float(request.args.get('pw'))

    # The features of the observation to predict
    features = [sepal_length,
                sepal_width,
                petal_length,
                petal_width]

    # Ladowanie modelu
    model = joblib.load('model.pkl')
    # Predykcja na podstawie modelu
    predicted_class = int(model.predict([features]))

    
    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(port=3333,host='0.0.0.0')