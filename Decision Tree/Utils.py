import numpy as np
import pandas as pd

def Gini(y):
    if type(y) is not pd.Series:
        classes = np.array([y])
        a = np.array([y])
        t = len(a)
    else:
        classes = y.unique()
        a = y.to_numpy()
        t = y.shape[0]
    val = 0
    for j in classes:
        j_count = np.count_nonzero(a == j)
        val += pow(j_count / t, 2)
    gini = 1 - val
    return gini

def Gini_split(ys):
    n = 0
    for y in ys:
        if not hasattr(y, 'shape'):
            n += 1
        else:
            n += 1 if len(y.shape) == 0 else y.shape[0]
    gs = 0
    for y in ys:
        if not hasattr(y, 'shape'):
            cant = 1
        else:
            cant = 1 if len(y.shape) == 0 else y.shape[0]
        gs += (cant * Gini(y)) / n
    return gs
   
def calculate_confusion_matrix(predict, real):
    confusion_matrix = np.zeros((len(predict.unique()), len(predict.unique())))
    indexes = predict.unique()
    for i in range(predict.shape[0]):
        if real.iloc[i] == predict.iloc[i]:
            index = np.where(indexes == real.iloc[i])
            index = index[0][0]
            confusion_matrix[index][index] += 1 
        elif real.iloc[i] in indexes:
            index1 = np.where(indexes == real.iloc[i])
            index1 = index1[0][0]
            index2 = np.where(indexes == predict.iloc[i])
            index2 = index2[0][0]
            confusion_matrix[index1][index2] += 1 

    return confusion_matrix

def calc_accuracy(confusion_matrix):
    numerator = 0
    denominator = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            if i == j: 
                numerator += confusion_matrix[i][j]
                denominator += confusion_matrix[i][j]
            else: 
                denominator += confusion_matrix[i][j]
                
    return numerator / denominator

