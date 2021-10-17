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

#Initilize data
np.random.seed(0)

maxdegrees = 8 #Max degree of polynomial fit
N = 20 #Total datapoints
ts = 0.2 #Size of train test split
scaling = True #Change to try without scaling

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y)
x_flat = np.ravel(x_mesh)
y_flat = np.ravel(y_mesh)

z = FrankeFunction(x_flat, y_flat, 0.15) #Change the third argument to vary the amount of stoicrastic noise

polydegree = np.zeros(maxdegrees)

mse_ols_train = np.zeros(maxdegrees)
mse_ols_test = np.zeros(maxdegrees)

# OLS regression for polynomials from 0 to maxdegrees
for i in range(maxdegrees):
    degree = i
    polydegree[i] = degree
    X = create_X(x_flat, y_flat, degree)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=ts)

    if (scaling == True):
        X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test) #Scale data with standard scaler

    # prediction ols
    beta_ols = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    zTilde_ols = X_train @ beta_ols
    zPredict_ols = X_test @ beta_ols

    # cost functions ols
    mse_ols_train[i] = MSE(z_train, zTilde_ols)
    mse_ols_test[i] = MSE(z_test, zPredict_ols)

print("Minimum MSE: ", np.amin(mse_ols_test))

plt.plot(polydegree, mse_ols_train, label='mse train')
plt.plot(polydegree, mse_ols_test, label='mse test')
plt.title("MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.legend()
plt.show()
