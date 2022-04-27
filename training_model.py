# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:29:18 2022

@author: micha
"""
from perceptron import Perceptron

import numpy as np
import copy
import pandas as pd
IRIS_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(IRIS_PATH, names=col_names)

simple_iris = copy.deepcopy(
    df[(
            df["class"].str.replace("Iris-","") == "setosa"
        ) | (
            df["class"].str.replace("Iris-","") == "versicolor"
        )]
    )
            
simple_iris.replace(
      to_replace=("Iris-setosa", "Iris-versicolor")
    , value = (-1,1), inplace = True
    )


X = copy.deepcopy(np.array(simple_iris.iloc[:,:4], dtype=np.float32))
y = copy.deepcopy(np.array(simple_iris.iloc[:,-1], dtype=np.float32))

model = Perceptron()
model.fit(X, y)

import pickle
import joblib
saved_model = pickle.dumps(model)
joblib.dump(model, 'model.pkl')




