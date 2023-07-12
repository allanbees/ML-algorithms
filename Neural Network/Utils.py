import pandas as pd
import numpy as np
from math import e

def load_data(file, drop_cols):
    df = pd.read_csv(file)
    df = df.drop(drop_cols, axis = 1)
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Sex'])
    y = np.array(df.Survived)
    y = np.reshape(y,(y.shape[0],1))
    df = ( df - df.mean()) / df.std()
    x = np.array(df.drop(['Survived'], axis = 1))
    return x, y

def error(y_pred, y_real):
    return np.sum((y_pred - y_real) ** 2)

def get_acc(y_pred, y_real):
    correct = 0
    samples = len(y_real)
    for i in range(len(y_pred)):
        correct += 1 if y_pred[i] == y_real[i] else 0
    return correct/samples

def init_activations(activation: list):
    act = d_act = out_act = d_out_act = 0
    if activation[0] == 'r':
        act = relu
        d_act = d_relu
    elif activation[0] == 'l':
        act = lrelu
        d_act = d_lrelu
    elif activation[0] == 't':
        act = tanh
        d_act = d_tanh
    else:
        act = sigmoid
        d_act = d_sigmoid
        
    if activation[1] == 'r':
        out_act = relu
        d_out_act = d_relu
    elif activation[1] == 'l':
        out_act = lrelu
        d_out_act = d_lrelu
    elif activation[1] == 't':
        out_act = tanh
        d_out_act = d_tanh
    else:
        out_act = sigmoid
        d_out_act = d_sigmoid
        
    return act, d_act, out_act, d_out_act

# Activation functions
def sigmoid(x): 
    return 1 / (1+ e ** -x)

def d_sigmoid(y): 
    return sigmoid(y) * (1 - sigmoid(y))

def tanh(x):
    return (e ** x - e ** -x) / (e**x + e ** -x)

def d_tanh(y): 
    return 1 - y ** 2

def relu(x): 
    return np.maximum(x, 0)

def d_relu(y):
    return (tanh(y) > 0) * 1

def lrelu(x): 
    return np.maximum(x, 0.01*x)

def d_lrelu(y): 
    x = np.zeros(y.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if y[i][j] > 0:
                x[i][j] = 1
            else: 
                x[i][j] = 0.01
    return x
