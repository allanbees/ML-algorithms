import numpy as np
import math
from Utils import error, init_activations
# recordar hacer ifs dependiendo del tipo de activacion que me manden 
class DenseNN: 
    
    def __init__(self, layers: list, activation: list, seed=21):
        self.layers = layers
        self.act, self.d_act, self.out_act, self.d_out_act = init_activations(activation)
        self.weights, self.b = self.init_xavier(seed)
        self.lr = 0
        self.epoch = 0
        self.dW = {}
        self.prev_dW = {}
        self.dB = {}
        self.dZ = {}
        self.Z = {}
        self.A = {}
        
    def predict(self, x):     
        _, A = self.forward_prop(x)
        return A[len(self.layers) - 1]  
    
    def forward_prop(self, x): 
        n = len(self.layers) - 1
        self.Z[1] = np.add(x.dot(self.weights[1]), self.b[1])
        self.A[1] = self.act(self.Z[1])
        for i in range(1,n):
            self.Z[i+1] = np.add(self.A[i].dot(self.weights[i+1]),self.b[i+1])
            if i+1 != n: 
                self.A[i+1] = self.act(self.Z[i+1])
            else:
                self.A[i+1] = self.out_act(self.Z[i+1])
        return self.Z, self.A
    
    def train(self, x, y, epochs=150000, lr=0.00001, momentum=0, decay=0.00000001):
        self.lr = lr
        self.momentum = momentum
        for epoch in range(epochs):
            self.forward_prop(x)    
            self.backpropagation(x, y)
            self.step()
            print(f"epoch [{epoch+1}/{epochs}], error: {error(self.predict(x), y)}")
            self.lr = self.lr / (1-decay)
    
    def backpropagation(self, x, y):
        num_layers = len(self.layers)
        for i in range(1, num_layers):
            li = num_layers - i
            li_m1 = li - 1
            aL = self.A[li]
            self.prev_dW[li] = self.dW[li] if self.epoch > 0 else 0 
            if li_m1 > 0: 
                aL_m1 = self.A[li_m1]
            if li == num_layers - 1:
                self.dZ[li] = 2 * (np.add(aL, -y))
                self.dW[li] = (np.dot(self.dZ[li].T, aL_m1)).T
                self.dB[li] = np.sum(self.dZ[li])
            elif li == 1:
                if self.weights[li + 1].shape[1] != self.dZ[li + 1].T.shape[1]:
                    self.dZ[li] = self.weights[li + 1].dot(self.dZ[li + 1].T) * self.d_act(self.Z[li].T)
                else:
                    self.dZ[li] = self.weights[li + 1].dot(self.dZ[li + 1]) * self.d_act(self.Z[li].T)
                self.dW[li] = (self.dZ[li].dot(x)).T 
                self.dB[li] = np.sum(self.dZ[li])
            else:
                if self.weights[li + 1].shape[1] != self.dZ[li + 1].T.shape[1]:
                    self.dZ[li] = self.weights[li + 1].dot(self.dZ[li + 1].T) * self.d_act(self.Z[li].T)
                else:
                    self.dZ[li] = self.weights[li + 1].dot(self.dZ[li + 1]) * self.d_act(self.Z[li].T)
                self.dW[li] = (self.dZ[li].dot(self.A[li_m1])).T
                self.dB[li] = np.sum(self.dZ[li])

            
    # updates weights and parameters
    def step(self):
        for i in range(1,len(self.layers)):
            self.weights[i] -= self.lr * self.dW[i] + self.momentum * self.prev_dW[i]
            self.b[i] -= self.lr * self.dB[i]
        self.epoch += 1
         
    def init_xavier(self, seed):
        weights = {}
        biases = {}
        np.random.seed(seed)
        for W in range(len(self.layers)-1):
            if W == 0:
                weights[W+1] = np.random.normal(0, 2.0/math.sqrt(5 + self.layers[W+1]), size = (self.layers[W], self.layers[W+1] ))
            else:
                weights[W+1] = np.random.normal(0, 2.0/math.sqrt(self.layers[W-1] + self.layers[W+1]), size = (self.layers[W],self.layers[W+1]))
            biases[W+1] = np.ones((1,self.layers[W+1]))
        return weights, biases

