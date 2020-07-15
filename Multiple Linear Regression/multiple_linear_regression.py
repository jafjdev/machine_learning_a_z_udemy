import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('50_Startups.csv')

# : toma todas las filas
# :-1 toma todas -1 , en este caso, toma todas
# las columnas menos la ultima, ya que es la
# variable independiente.
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 4].values

# ¿Datos incompletos?

from sklearn.impute import SimpleImputer

# En los valores que no se encuentren ,
# se le aplica el metodo de la media.
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])  # solo toma las columnas 1 y 2
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Encoding de categorical Variables
ct = ColumnTransformer(transformers=[('State',  # Nombre de la fila a la cual se le quiere aplicar
                                      OneHotEncoder(),  # transformador
                                      [3])],  # columna a la que es aplicada el transformador
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Evitando la trampa de la dummy variable
X = X[:, 1:]

# Splitting dataset en training y test
from sklearn.model_selection import train_test_split

# se le envia la matriz X, y la matriz Y,
# y el numero  de valores que se le asignaran
# al test_size,  0.2 representa el 20%
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# predicciendo

y_pred = regressor.predict(X_test)

# backward elimination

import statsmodels.api as sm

# se agrega una columna de 1 ya que la libreria de MLR no toma e cuenta

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# SL = 0.0.5 --- 5% .P.
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)

regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 4, 5]]
X_opt = np.array(X_opt, dtype=float)
# SL = 0.0.5 --- 5%
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()

X_opt = X[:, [0, 3, 5]]
X_opt = np.array(X_opt, dtype=float)
# SL = 0.0.5 --- 5%
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()

X_opt = X[:, [0, 3]]
X_opt = np.array(X_opt, dtype=float)
# SL = 0.0.5 --- 5%
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()
