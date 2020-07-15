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

from sklearn.preprocessing import Imputer

imputer = Imputer
