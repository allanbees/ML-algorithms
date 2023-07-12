# Allan Barrantes B80986

from LinearRegression import LR
from Functions import load_data, score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X, y = load_data('fish_perch.csv', 'Weight')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
my_lr = LR()
error, W = my_lr.fit(X_train, y_train, 500, 0.1e-4, 0.1e-3, 0.1e-4, 0.1e-10, 'mse')
y_predict = my_lr.predict(X_test)

print("\nMy linear regression")
print(f"The error is: {error}")
print(f"r2 is: {r2_score(y_test, y_predict)}")
print(f"my_r2 is: {score(y_test, y_predict)}")

# sklearn linear regression
print("\nSklearn linear regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(f"r2 is: {r2_score(y_test, pred)}")
print(f"my_r2 is: {score(y_test, pred)}")

