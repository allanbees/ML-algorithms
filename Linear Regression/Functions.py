import numpy as np
import pandas as pd
import math

def load_data(csv_file, target):
    df = pd.read_csv(csv_file)
    y = df[target]
    x = df.drop(columns=[target])
    x.insert(0, 'bias', 1)
    return x, y
    
def MSE(y_true, y_predict):
    n = y_true.shape[0]
    return np.sum( (y_true - y_predict) ** 2) / n

def MAE(y_true, y_predict):
    n = y_true.shape[0]
    return np.sum( abs(y_true - y_predict) )/n

def RMSE(y_true, y_predict):
    return math.sqrt(MSE(y_true, y_predict))

def score(y_true, y_predict):
    return 1 - sum((y_true - y_predict) **2 ) / sum((y_true - (sum(y_true)/y_true.shape[0])) ** 2)