'''
Author: Andreas Dyve, Jørn-Marcus Høylo-Rosenberg
Date: october 2021
'''

'''
toDo:
- Plot som funksjon av n_epochs
- Tuning av eta, M
- Samme med ridge, heat map av lambda og eta
'''

import autograd.numpy as np
from autograd import elementwise_grad
from module1 import Franke
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

def MSE(beta):
    return ((X_train@beta - z_train)**2)/np.size(z_train)

def schedule(eta, decay, epoch):
    return eta/(1 + decay*epoch)

#From https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
def create_miniBatches(X,y, M):
    mini_batches = []
    data = np.hstack((X, y.reshape(-1,1)))
    np.random.shuffle(data)
    m = data.shape[0] // M
    i = 0

    for i in range(m + 1):
        mini_batch = data[i * M:(i + 1)*M, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1]
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % M != 0:
        mini_batch = data[i * M:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1]
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

# Initialise data
np.random.seed(64)
N = 100
polydegree = 5
noise_coef = 0.15
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()
print("MSE scores")
"""
# Regular OLS
beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
ytilde = X_train @ beta
print("Analytical", mean_squared_error(z_train, ytilde))

#Gradient descent
np.random.seed(64)
max_iter = 10000
eta = 0.05
iter = 0
beta_gd = np.random.randn(X_train.shape[1])
while iter < max_iter:
    gradient = (2.0/len(X_train))*X_train.T @ (X_train @ beta_gd - z_train)
    beta_gd = beta_gd - eta*gradient
    iter += 1
ytilde_gd = X_train @ beta_gd
print("GD ", mean_squared_error(z_train, ytilde_gd))
"""
#SGD Without timeschedule
np.random.seed(64)
n_epochs = 100
eta = 0.001
M = 5
theta = np.random.randn(X_train.shape[1])
for epoch in range(n_epochs):
    mini_batches = create_miniBatches(X_train, z_train, M)
    for mini_batch in mini_batches:
        xi, yi = mini_batch
        gradient = 2.0*xi.T@((xi@theta)-yi)
        theta = theta - eta*gradient
ytilde_sdg = X_test @ theta
mse_sdg = mean_squared_error(z_test, ytilde_sdg)
print("SGD without timeschedule ", mse_sdg)
"""
#SGD With timeschedule
np.random.seed(64)
n_epochs = 100
M = 5
m = int(len(X_train)/M)
eta_ls = 0.01
decay = 1e-6

theta_ls = np.random.randn(X_train.shape[1])
for epoch in range(n_epochs):
    mini_batches = create_miniBatches(X_train, z_train, M)
    k = 0
    for mini_batch in mini_batches:
        xi, yi = mini_batch
        gradient_ls = 2.0*xi.T@((xi@theta_ls)-yi)
        eta_ls = schedule(eta_ls, decay, epoch)
        theta_ls = theta_ls - eta_ls*gradient_ls
        k+=1
ytilde_sdg_ls = X_train @ theta_ls
mse_sdg_ls = mean_squared_error(z_train, ytilde_sdg_ls)
print("SGD with timeschedule ", mse_sdg_ls)

#RMSPROP
np.random.seed(64)
n_epochs = 100
M = 5
eta = 0.001
m = int(len(X_train)/M)
theta = np.random.randn(X_train.shape[1])
#s = np.random.randn(X_train.shape[1]) #gir negative tall, så kvadratroten i første iterasjon gir NaN
#s = np.random.normal(1,0.15,X_train.shape[1]) #Funker
s = np.zeros(X_train.shape[1]) #Funker
delta = 0.9
eps = 1e-8
for epoch in range(n_epochs):
    mini_batches = create_miniBatches(X_train, z_train, M)
    for mini_batch in mini_batches:
        xi,yi = mini_batch
        gradients = 2 * xi.T @ ((xi @ theta)-yi)
        g = gradients*gradients
        s = delta*s + (1-delta)*g
        theta = theta - (eta/np.sqrt(s + eps))*gradients
ytilde_sdg_copy = X_train @ theta
mse_sdg_copy_rmsprop = mean_squared_error(z_train, ytilde_sdg_copy)
print("SDG with RMSPROP algo ", mse_sdg_copy_rmsprop)
"""
