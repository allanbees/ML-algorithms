from Utils import Gini_split
import numpy as np
import pandas as pd

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self._type = ''
        self._class = ''
        self.gini = 1
        self.count = 0
        self.split_type = ''
        self.split_column = ''
        self.split_value = ''
        self.depth = 0
    
    def to_dict(self):
        n_dict = {}
        n_dict['type'] = self._type
        if self._type == 'Leaf':
            n_dict['class'] = self._class
            n_dict['count'] = self.count
        elif self._type == 'Split':
            n_dict['gini'] = self.gini
            n_dict['split_type'] = self.split_type
            n_dict['split_column'] = self.split_column
            n_dict['split_value'] = self.split_value
        return n_dict
    
class DecisionTree:
    def __init__(self, target):
        self.root = None
        self.target = target
        
     
    def fit(self, x, y, max_depth = None):
        if max_depth and max_depth > 0:
            self.fit_r(x, y, max_depth, self.root)
        
    def fit_r(self, x, y, max_depth, node):
        if node == None:
            node = Node()
            node._type = 'Split'
            self.root = node   
        x1 = None
        x2 = None
        y1 = None
        y2 = None
        num_cols = x.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        for col in x:
            if col == self.target:
                continue
            series = x[col].squeeze()
            if col in num_cols:
                smin = series.min()
                smax = series.max()
                if smin == smax:
                    continue
                points = np.linspace(smin, smax, 10)
                for p in points:
                    c1 = x[x[col] <= p]
                    c2 = x[x[col] > p]
                    s1 = c1[self.target].squeeze()
                    s2 = c2[self.target].squeeze()
                    g_s = Gini_split([s1, s2])
                    if g_s < node.gini:
                        node.split_type = 'Numerical'
                        node.gini = g_s
                        node.split_column = col
                        node.split_value = p
                        x1 = c1
                        x2 = c2
                        y1 = s1
                        y2 = s2
            else:
                node.split_type = 'Categorical'
                classes = series.unique()
                for j in classes:
                    # Divide df in j and not j
                    c1 = x[x[col] == j]
                    c2 = x[x[col] != j]
                    s1 = c1[self.target]
                    s2 = c2[self.target]
                    g_s = Gini_split([s1, s2])
                    if g_s < node.gini:
                        node.gini = g_s
                        node.split_column = col
                        node.split_value = j
                        x1 = c1
                        x2 = c2
                        y1 = s1
                        y2 = s2
                        
        if node.gini == 0 or node.depth == max_depth-1:
            Lnode = Node()
            Lnode._type = 'Leaf'
            Lnode.depth = node.depth + 1
            if hasattr(y1, 'unique'):
                classes1 = y1.unique()
            else:
                classes1 = [ y1 ]
            if len(classes1) > 1: 
                a = y1.to_numpy()
                max_j = ''
                max_cant = 0
                for j in classes1:
                    j_count = np.count_nonzero(a == j)
                    if j_count > max_cant:
                        max_j = j
                        max_cant = j_count
                Lnode._class = max_j
            elif len(classes1) == 1:
                Lnode._class = classes1[0]
            if hasattr(y1, 'shape') and len(y) > 0:
                Lnode.count = y1.shape[0]
            else:
                Lnode.count = 1
            node.left = Lnode
            Rnode = Node()
            Rnode._type = 'Leaf'
            Rnode.depth = node.depth + 1
            if hasattr(y2, 'unique'):
                classes2 = y2.unique()
            else:
                classes2 = [ y2 ]
            if len(classes2) > 1: 
                a = y2.to_numpy()
                max_j = ''
                max_cant = 0
                for j in classes2:
                    j_count = np.count_nonzero(a == j)
                    if j_count > max_cant:
                        max_j = j
                        max_cant = j_count
                Rnode._class = max_j
            elif len(classes2) == 1:
                Rnode._class = classes2[0]
            if hasattr(y2, 'shape'):
                Rnode.count = y2.shape[0]
            else:
                Rnode.count = 1
            node.right = Rnode 
        else:
            Lnode = Node()
            Lnode.depth = node.depth + 1
            Lnode._type = 'Split'
            node.left = Lnode
            Rnode = Node()
            Rnode.depth = node.depth + 1
            Rnode._type = 'Split'
            node.right = Rnode
            self.fit_r(x1, y1, max_depth, Lnode)
            self.fit_r(x2, y2, max_depth, Rnode)

            
    def predict(self, x):
        data_dict = {}
        for index, row in x.iterrows():
            current_node = self.root
            while current_node != None:
                if current_node._type == 'Leaf':
                    data_dict[index] = current_node._class
                    current_node = None
                else:
                    if current_node._type == 'Split':
                        current_node = current_node.left if row[current_node.split_column] == current_node.split_value else current_node.right
                    else:
                        current_node = current_node.left if row[current_node.split_column] <= current_node.split_value else current_node.right
        return pd.Series(data=data_dict)
    
    def to_dict(self):
        return self.to_dict_r(self.root)
        
    def to_dict_r(self, node):
        if node == None:
            return "No existen nodos"
        _dict = node.to_dict()
        if node.left != None:
            L_dict = self.to_dict_r(node.left)
            _dict['child-left'] = L_dict
        if node.right != None:
            R_dict = self.to_dict_r(node.right)
            _dict['child-right'] = R_dict
        return _dict
