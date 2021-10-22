import numpy as np
import matplotlib.pyplot as plt
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from module1 import Sdg, Franke

np.random.seed(64)
N = 20
polydegree = 5
noise_coef = 0
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

eta = 0.015
M = 6
n_epochs = 100

sgdRegressor = Sdg(X_train, X_test, z_train, z_test, eta, M, n_epochs)
sgdRegressor.stocastichGD_ols()
sgdRegressor.stocastichGD_ridge(1e-2)
