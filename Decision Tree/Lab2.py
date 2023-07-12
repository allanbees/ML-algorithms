# =============================================================================
# Estudiantes: Allan Barrantes B80986, Javier Sandoval B56762
# =============================================================================


import pandas as pd
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree
from Utils import calculate_confusion_matrix, calc_accuracy
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


# print("Mushrooms decision tree")
# print("-----------------------------------------------------------------\n")
# x = pd.read_csv('mushrooms.csv')
# x = x.dropna()
# y = x['class']
# tree = DecisionTree('class')
# tree.fit(x, y, 6)
# print(tree.to_dict())
# p = tree.predict(x)
# confusion_matrix = calculate_confusion_matrix(p, y)
# print()
# print("Confusion matrix: ")
# print(confusion_matrix)
# print()
# print("Confusion matrix using sklearn")
# conf_matrix_sklearn = metrics.confusion_matrix(y, p, labels=p.unique())
# print(conf_matrix_sklearn)
# print(f"Accuracy is: {calc_accuracy(confusion_matrix)}")

# print("Titanic decision tree")
# print("-----------------------------------------------------------------\n")
# x = pd.read_csv('titanic.csv')
# x = x.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)
# x = x.dropna()
# y = x['Survived'].squeeze()
# tree = DecisionTree('Survived')
# tree.fit(x, y, 4)
# print(tree.to_dict())
# p = tree.predict(x)
# confusion_matrix = calculate_confusion_matrix(p, y)
# print()
# print("Confusion matrix: ")
# print(confusion_matrix)
# print()
# print("Confusion matrix using sklearn")
# conf_matrix_sklearn = metrics.confusion_matrix(y, p, labels=p.unique())
# print(conf_matrix_sklearn)
# print(f"Accuracy is: {calc_accuracy(confusion_matrix)}")

print("Iris decision tree")
print("-----------------------------------------------------------------\n")
x = pd.read_csv('iris.data', names=['sepal-length','sepal-width','petal-length','petal-width','class'])
x = x.dropna()
y = x['class'].squeeze()
tree = DecisionTree('class')
tree.fit(x, y, 5)
print(tree.to_dict())
p = tree.predict(x)
confusion_matrix = calculate_confusion_matrix(p, y)
print()
print("Confusion matrix: ")
print(confusion_matrix)
print()
print("Confusion matrix using sklearn")
conf_matrix_sklearn = metrics.confusion_matrix(y, p, labels=p.unique())
print(conf_matrix_sklearn)
print(f"Accuracy is: {calc_accuracy(confusion_matrix)}")


# sklearn decision tree mushrooms
# df = pd.read_csv('mushrooms.csv')
# y = df['class']
# x = df.drop(columns=['class'])

# for col in x.columns:
#     x[col] = x[col].astype('category')
#     mapping = x[col].cat.categories
#     x[col] = x[col].cat.codes

# y = y.astype('category')
# mapping = y.cat.categories
# y = y.cat.codes

# X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)
# tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth= 3 )
# tree.fit(X_train,y_train)
# prediction = tree.predict(X_test)
# fig = plt.figure(figsize=(25,20))
# plot_tree(tree, feature_names=X_train.columns,class_names=["P", "E"], filled=True)
# plt.show()


# sklearn decision tree titanic
# df = pd.read_csv('titanic.csv')
# df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']).dropna()
# y = df['Survived']
# x = df.drop(columns=['Survived'])
# x['Pclass'] = x['Pclass'].astype('category')
# x['Sex'] = x['Sex'].astype('category')

# for col in x.columns:
#     x[col] = x[col].astype('category')
#     mapping = x[col].cat.categories
#     x[col] = x[col].cat.codes

# y = y.astype('category')
# mapping = y.cat.categories
# y = y.cat.codes

# X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)
# tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth= 3 )
# tree.fit(X_train,y_train)
# prediction = tree.predict(X_test)
# fig = plt.figure(figsize=(25,20))
# plot_tree(tree, feature_names=X_train.columns,class_names=["Survived", "Died"], filled=True)
# plt.show()


# sklearn decision tree iris
x = pd.read_csv('iris.data', names=['sepal-length','sepal-width','petal-length','petal-width','class'])
y = x['class']
x = x.drop(columns=['class'])

y = y.astype('category')
mapping = y.cat.categories
y = y.cat.codes

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=0)
tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth= 1 )
tree.fit(X_train,y_train)
prediction = tree.predict(X_test)
fig = plt.figure(figsize=(25,20))
plot_tree(tree, feature_names=X_train.columns,class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"], filled=True)
plt.show()


