import numpy as np
import matplotlib.pyplot as plt
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from module1 import Sdg, Franke
from sklearn.linear_model import SGDRegressor


def ols_regression(X_train, y_train):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    ytilde= X_train @ beta
    mse = mean_squared_error(y_train, ytilde)
    print("OLS")
    print(mse)

def ridge_regression(X_train, y_train, lmd):
     I_ = np.eye(X_train.shape[1], X_train.shape[1])
     beta = np.linalg.pinv(X_train.T @ X_train + lmd*I_) @ X_train.T @ y_train
     ytilde = X_train @ beta
     mse = mean_squared_error(y_train, ytilde)
     print("ridge")
     print(mse)

np.random.seed(64)
N = 20
polydegree = 2
noise_coef = 0
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

eta = 0.015
M = 5
n_epochs = 100
m = int(len(X_train)/M)

sgdRegressor = Sdg(X_train, X_test, z_train, z_test, eta, M, n_epochs)
j = 0
while j < 50:
    #sgdRegressor.grad_descent(1000)
    sgdRegressor.stocastichGD_ols()
    #sgdRegressor.stocastichGD_ridge(1)
    j+=1
