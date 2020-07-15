import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Salary_Data.csv')
# : toma todas las filas
# :-1 toma todas -1 , en este caso, toma todas
# las columnas menos la ultima, ya que es la
# variable independiente.
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 1].values

# Splitting dataset en training y test
from sklearn.model_selection import train_test_split

# se le envia la matriz X, y la matriz Y,
# y el numero  de valores que se le asignaran
# al test_size,  0.2 representa el 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Fitting SLR(Simple Linear Regression) to the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediciendo los valores de las matrices de train

y_pred = regressor.predict(X_test)  # se predicen los valores del test set

# Graficando

plt.scatter(X_test, Y_test)
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary Vs Experience Test set')
plt.xlabel("User Experience")
plt.ylabel("Salary")
plt.show()
