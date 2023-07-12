import numpy as np
import math
from numpy import linalg as LA

class myPCA:

    def center_reduce(self, data):
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        return (data - mean) / std
    
    def corr_points(self, V, eigenvalues):
        corr_points = np.zeros((V.shape[0], 2))
        for i in range(V.shape[0]):
            corr_points[i][0] = V[i][0] * math.sqrt(eigenvalues[0])
            corr_points[i][1] = V[i][1] * math.sqrt(eigenvalues[1])
        return corr_points
    
    def get_pca( self, np_data ):
        reduced = self.center_reduce(np_data)
        R =  1/reduced.shape[0] * np.matmul(reduced.transpose(), reduced)
        eigenvalues, eigenvectors = LA.eigh(R)
        eigenvalues = np.abs(eigenvalues)
        ind = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[ind]
        V = eigenvectors[:,ind]
        C = np.matmul(reduced, V)
        inertia = eigenvalues / reduced.shape[1]
        corr_pts = self.corr_points(V, eigenvalues) 
        return C, inertia, corr_pts