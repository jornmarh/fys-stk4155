# Import librarie/packages
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import SGDRegressor

#Define functions
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y,noise_coef):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    noise = noise_coef * np.random.randn(N*N)
    return term1 + term2 + term3 + term4 + noise

def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of columns in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def scale(X_train, X_test, z_train, z_test):
    scaler = StandardScaler() # Utilizing scikit's standardscaler

    scaler_x = scaler.fit(X_train) # Scaling x-data
    X_train_scaled = scaler_x.transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_z = scaler.fit(z_train.reshape(-1,1)) # Scaling z-data
    z_train_scaled = scaler_z.transform(z_train.reshape(-1,1)).ravel()
    z_test_scaled = scaler_z.transform(z_test.reshape(-1,1)).ravel()

    return X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled

def learning_schedule(t):
    return t_0/(t + t_1)

#Initilize data
np.random.seed(0)
N = 20 #Total datapoints

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y)
x_flat = np.ravel(x_mesh)
y_flat = np.ravel(y_mesh)
z = FrankeFunction(x_flat, y_flat, 0) #Change the third argument to vary the amount of stoicrastic noise


X = create_X(x_flat, y_flat, 5)
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test)

beta_ols = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
print("Beta: OLS")
print(beta_ols.reshape(-1,1))

#paramateres
M = 10 #Number of points per mini-batch
m = int(len(X_train)/M) #Amount of mini-batches
n_epochs = 100 # Number of total epochs

#paramateres for tuning the learning rate
t_0 = 5
t_1 = 100
eta = t_0/t_1
#Initial guess for the paramater beta(theta)
theta = np.random.randn(len(X_train[0,:]))

#SDG algorithm
algo = "Basic"

for epoch in range(1, n_epochs+1):
    for k in range(m):
        random_index = np.random.randint(m)
        xi = X_train[random_index:random_index+1]
        zi = z_train[random_index:random_index+1]
        if (algo == "Basic"):
            eta = learning_schedule(epoch*m+k)
        grad = 2.0*xi.T@((xi@theta)-zi)
        theta = theta - eta*grad
print("SGD")
print(np.abs(theta - beta_ols))

sgdreg = SGDRegressor(max_iter = 100, penalty=None, eta0=t_0/t_1)
sgdreg.fit(X_train,z_train)
print("sgdreg from scikit")
print(np.abs(sgdreg.coef_ - beta_ols))
