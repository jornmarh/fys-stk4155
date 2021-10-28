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
N = 100
polydegree = 5
noise_coef = 0.15
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

eta = 0.001
alpha = 0.01
M = 5
n_epochs = 100
m = int(len(X_train)/M) #8000/M

epoch_numbers = np.arange(30)
mse = np.zeros(len(epoch_numbers))
i = 0
for epoch_number in epoch_numbers:
    sgdRegressor = Sdg(X_train, X_test, z_train, z_test, eta, alpha, M, epoch_number)
    #sgdRegressor.grad_descent(1000)
    mse[i] = sgdRegressor.stocastichGD_ols('rmsprop')
    #sgdRegressor.stocastichGD_ridge(0,10)
    i+=1
plt.plot(epoch_numbers, mse, label='lr: %.4f, m: %i' %(sgdRegressor.eta, sgdRegressor.m))
plt.xlabel('Number of epochs')
plt.ylabel('mse')
plt.legend()
plt.ylim(0,1)
plt.show()
