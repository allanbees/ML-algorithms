"""
Lab 4
Allan Barrantes, Javier Sandoval
"""
from Utils import load_data, get_acc
from NeuralNetwork import DenseNN
from sklearn.model_selection import train_test_split

x, y = load_data('titanic.csv', ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
nn = DenseNN([6,5,2,1], ['r', 's'], 21) 

nn.train(X_train, y_train)
y_pred=nn.predict(X_test)

y_pred, y_test = y_pred.squeeze(), y_test.squeeze()
for i in range(len(y_pred)):
    y_pred[i] = 1 if y_pred[i] > 0.5 else 0

print(f"Accuracy de la red es {get_acc(y_pred,y_test)}")