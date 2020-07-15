import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Data.csv')
# : toma todas las filas
# :-1 toma todas -1 , en este caso, toma todas
# las columnas menos la ultima, ya que es la
# variable independiente.
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 3].values

# ¿Datos incompletos?

from sklearn.impute import SimpleImputer

# En los valores que no se encuentren ,
# se le aplica el metodo de la media.
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])  # solo toma las columnas 1 y 2
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Encoding de categorical Variables
ct = ColumnTransformer(transformers=[('Country',  # Nombre de la fila a la cual se le quiere aplicar
                                      OneHotEncoder(),  # transformador
                                      [0])],  # columna a la que es aplicada el transformador
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# No categoricas
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# Splitting dataset en training y test
from sklearn.model_selection import train_test_split

# se le envia la matriz X, y la matriz Y,
# y el numero  de valores que se le asignaran
# al test_size,  0.2 representa el 20%
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""# Feature Scaling   (Muchas librerias hacen este trabajo, no se tiene que hacer manualmente)
from sklearn.preprocessing import StandardScaler

standard_scaler_X = StandardScaler()
X_train = standard_scaler_X.fit_transform(X_train)
# Importante siempre escalar la matriz
# de entrenamiento, con el metodo de fit & transform y de esta manera,
# la matriz de test, estara escalada en la misma escala
X_test = standard_scaler_X.transform(X_test)

# No es necesario escalar la matriz Y, ya que los valores de esta,
# estan entre 0 y 1 , no varian considerablemente."""
