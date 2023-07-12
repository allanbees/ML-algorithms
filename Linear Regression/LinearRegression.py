import numpy as np
from Functions import MSE, MAE

class LR: 
   
    def __init__(self):
        self.W = 0
        
    def fit(self, x, y, max_epochs=100000, threshold=0.1e-4, learning_rate=0.1e-3, momentum=0.1e-4, decay=0.1e-10, error='mse', regularization='none', _lambda=0):
        m = x.shape[1]
        n = y.shape[0]
        err = 0
        prev_error = 0 
        prev_dW = 0
        dW = 0
        self.W = np.random.randn(m)
        
        for i in range(max_epochs):       
            prev_error = err
            y_predict = np.dot(x, self.W.T)
            if error == 'mse':    
                err = MSE(y, y_predict)
            else:
                err = MAE(y, y_predict)
            prev_dW = dW
            dW = 2 * (np.dot((y_predict - y).T, x )).T / n
            if regularization == 'l1' or regularization == 'lasso':
                err += sum( abs(self.W) ) * _lambda 
                dW += _lambda
            elif regularization == 'l2' or regularization == 'ridge':
                err += sum( abs(self.W) ** 2 ) * _lambda
                dW += 2 * _lambda * self.W
            elif regularization == 'elastic-net':
                err += _lambda * sum(self.W ** 2) + (1 - _lambda) * sum(abs(self.W))
                
            if abs(prev_error - err) < threshold:
                print(f"Epoch {i}/{max_epochs} with final error of {err}")
                break
            if i % 1000 == 0:
                print(f"Epoch {i}/{max_epochs}, error = {err}")
            self.W -=  learning_rate * (dW + momentum * prev_dW) 
            learning_rate = learning_rate / (1 + decay)
        return err, self.W
    
    def predict(self, x):
        return x.dot(self.W)
