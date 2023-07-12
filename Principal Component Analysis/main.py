import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from PCA import myPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# Decidí remover las columnas: PassengerId, Name, Ticket, Fare, Cabin, Embarked
# No escogí esas porque me parecieron columnas con datos que no aportan información 
# valiosa a la hora de determinar si una persona sobrevivió o no, por ejemplo de nada 
# me sirve saber el nombre de una persona o el puerto donde embarcó.
# =============================================================================

def load_dataset( data ):
    df = pd.read_csv(data, usecols = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
    df = pd.get_dummies(df, columns=['Pclass','Sex'])
    df = df.dropna() 
    df = df.drop(columns=['Sex_male', 'Pclass_2'])
    return df
    
# Método para generar gráficos de dispersión y círculo de correlación
def generatePlots(data, C, V, inertia): 
    # 1. Scatter plot
    plt.scatter(np.ravel(C[:,0]), np.ravel(C[:,1]), c = ['b' if i == 1 else 'r' for i in data['Survived']])
    plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
    plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[1],))
    plt.title('PCA')
    plt.show()
    
    # 2. Correlation circle
    plt.figure(figsize=(15, 15))
    plt.axhline(0, color='b')
    plt.axvline(0, color='b')
    for i in range(0, data.shape[1]):
        plt.arrow(0,0, V[i][0], V[i][1], head_width = 0.05, head_length = 0.05)
        plt.text(V[i][0] + 0.05, V[i][1] + 0.05, data.columns.values[i])
    an = np.linspace(0,2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an), color="b")
    plt.axis('equal')
    plt.title('Correlation circle')
    plt.show() 

# myPCA 
pca = myPCA()
data = load_dataset("titanic.csv") # pandas df
data_np = np.array(data) # df to numpy array
C, inertia, corr_pts = pca.get_pca(data_np)
generatePlots(data, C, corr_pts, inertia)

# sklearn pca
scaler = StandardScaler(copy=True,with_mean=True, with_std=True)
scaled_data = scaler.fit_transform(data)
pca = PCA()
C = pca.fit_transform(scaled_data)
inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(scaled_data.shape[1]))
generatePlots(data, C, V, inertia)

# Tabla de correlación
corrMatrix = data.corr()
sn.heatmap(corrMatrix, annot = True)











