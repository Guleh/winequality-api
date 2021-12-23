from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score

app = FastAPI()

@app.get("/")
def index(fixed_acidity = 0, volatile_acidity = 0, citric_acid = 0,
               residual_sugar = 0, chlorides = 0, free_sulfur_dioxide = 0,
               total_sulfur_dioxide = 0, density = 0, pH = 0, sulphates = 0,
               alcohol = 0):
    result = makemodel('extratrees', fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
           alcohol)
    return result

@app.get("/extratreesregressor/")
def index(fixed_acidity = 0, volatile_acidity = 0, citric_acid = 0,
               residual_sugar = 0, chlorides = 0, free_sulfur_dioxide = 0,
               total_sulfur_dioxide = 0, density = 0, pH = 0, sulphates = 0,
               alcohol = 0):
    result = makemodel('extratrees', fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
           alcohol)
    return result

@app.get("/randomforestclassifier/")
def index(fixed_acidity = 0, volatile_acidity = 0, citric_acid = 0,
               residual_sugar = 0, chlorides = 0, free_sulfur_dioxide = 0,
               total_sulfur_dioxide = 0, density = 0, pH = 0, sulphates = 0,
               alcohol = 0):
    result = makemodel('randomforest', fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
           alcohol)
    return result

def makemodel(model, fixed_acidity = 0, volatile_acidity = 0, citric_acid = 0,
               residual_sugar = 0, chlorides = 0, free_sulfur_dioxide = 0,
               total_sulfur_dioxide = 0, density = 0, pH = 0, sulphates = 0,
               alcohol = 0):
    result = get_result(model, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
           alcohol)
    if model == 'extratrees':
        model = 'Extra Trees regressor'
    if model == 'randomforest':
        model = 'Random Forest classifier'
    data = {'predicted_quality' : result[0],
            'model accuracy' : result[1],
            'model_used' : model,
            'fixed_acidity' : fixed_acidity,
            'volatile_acidity' : volatile_acidity,
            'citric_acid' : citric_acid,
            'residual_sugar' : residual_sugar,
            'chlorides' : chlorides,
            'free_sulfur_dioxide' : free_sulfur_dioxide,
            'total_sulfur_dioxide' : total_sulfur_dioxide,
            'density' : density,
            'pH' : pH,
            'sulphates' : sulphates,
            'alcohol' : alcohol}
    return data

def get_dataset():
    return pd.read_csv('winequality-white.csv', sep=';')

def preprocess(dataset):
    Q1 = dataset['volatile acidity'].quantile(0.25)
    Q3 = dataset['volatile acidity'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataset = dataset[(dataset['volatile acidity']>lower_limit)&(dataset['volatile acidity']<upper_limit)]
    Q1 = dataset['residual sugar'].quantile(0.25)
    Q3 = dataset['residual sugar'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataset = dataset[(dataset['residual sugar']>lower_limit)&(dataset['residual sugar']<upper_limit)]
    Q1 = dataset['chlorides'].quantile(0.25)
    Q3 = dataset['chlorides'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataset = dataset[(dataset['chlorides']>lower_limit)&(dataset['chlorides']<upper_limit)]
    Q1 = dataset['total sulfur dioxide'].quantile(0.25)
    Q3 = dataset['total sulfur dioxide'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 + 1.5*IQR
    dataset = dataset[(dataset['total sulfur dioxide']>lower_limit)&(dataset['total sulfur dioxide']<upper_limit)]
    return dataset


def get_result(model, fixed_acidity, volatile_acidity, citric_acid,
               residual_sugar, chlorides, free_sulfur_dioxide,
               total_sulfur_dioxide, density, pH, sulphates,
               alcohol):

    dataset = preprocess(get_dataset())
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)
    sc_x = RobustScaler(quantile_range=(25.0, 75.0))
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    if model == 'extratrees':
        model = ExtraTreesRegressor(random_state = 1)
    if model == 'randomforest':
        model = RandomForestClassifier(random_state = 1)
    model.fit(x_train, y_train)
    y_pred = np.round(model.predict(x_test))
    accuracy = round(accuracy_score(y_pred, y_test), 4)
    row = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
           chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
           alcohol]]
    row = sc_x.transform(row)
    y_pred = model.predict(row)
    return [str(y_pred[0]), accuracy]
