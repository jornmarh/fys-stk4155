import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.utils import resample
from sklearn.model_selection import KFold


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

def ols(X_train, X_test, z_train, z_test, output):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    z_tilde = X_train @ beta
    z_predict = X_test @ beta
    if (output == 1): return MSE(z_train, z_tilde), MSE(z_test, z_predict)
    if (output == 2): return R2(z_train, z_tilde), R2(z_test, z_predict)
    if (output == 3): return mean_squared_error(z_train, z_tilde), mean_squared_error(z_test, z_predict)

def ridge(X_train, X_test, z_train, z_test, _lambda):
    _I = np.eye(X_train.shape[1], X_train.shape[1])
    beta= np.linalg.pinv(X_train.T @ X_train + _lambda*_I) @ X_train.T @ z_train
    z_tilde = X_train @ beta
    z_predict = X_test @ beta
    return MSE(z_train, z_tilde), MSE(z_test, z_predict)

def scale(X_train, X_test, z_train, z_test):
    scaler = StandardScaler() #Utilizing scikit's standardscaler

    scaler_x = scaler.fit(X_train) #Scaling x-data
    X_train_scikit = scaler_x.transform(X_train)
    X_test_scikit = scaler_x.transform(X_test)

    scaler_z = scaler.fit(z_train.reshape(-1,1)) #Scaling z-data
    z_train_scikit = scaler_z.transform(z_train.reshape(-1,1)).ravel()
    z_test_scikit = scaler_z.transform(z_test.reshape(-1,1)).ravel()

    return X_train_scikit, X_test_scikit, z_train_scikit, z_test_scikit

#Initilize data
seed(42)
n = 5
maxdegree = 11
N = 20
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
z = FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N)

#Polynomial analysis
complexity = False #Change to True to perform simple analysis of polynomial degree
e = np.zeros(maxdegree)
b = np.zeros(maxdegree)
v = np.zeros(maxdegree)

#Bootstrap
bootstrap = True #Change to True to perform bootstrap analysis for various polynomial degrees
nBootstrap = 1000 #Number of bootstraps
error_ols_bootstrap = np.zeros(maxdegree)
bias_ols_bootstrap = np.zeros(maxdegree)
var_ols_bootstrap = np.zeros(maxdegree)

#Cross-validation
cvd = True #Change to True to perform cross validation analysis for various polynomial degrees.
k = 10
error_ols_cvd = np.zeros(maxdegree)

#Complexity analysis
degrees = np.zeros(maxdegree)
for i in range(maxdegree):
    #Crate design matrix for every degree until maxdegree
    degrees[i] = i+1
    X = create_X(xflat, yflat, i)
    print("\n", "Degree: ", degrees[i])

    #Perform cvd analysis
    if(cvd == True):
        kfold = KFold(n_splits=k, shuffle=True) #Use the KFold split method from Scikit-learn
        split=0 #Variable to keep track of the specific split
        error_ols_cvd_split = np.zeros(k) #Array of errors for every split
        print("CVD")
        for train_inds, test_inds in kfold.split(X):
            #Split and scale data
            xtrain = X[train_inds]; ztrain = z[train_inds]
            xtest = X[test_inds]; ztest = z[test_inds]
            Xtrain, Xtest, ztrain, ztest = scale(xtrain, xtest, ztrain, ztest)

            #OLS prediction
            beta_cvd = np.linalg.pinv(Xtrain.T @ Xtrain) @ Xtrain.T @ ztrain
            zPredict_cvd = Xtest @ beta_cvd
            #Print error per split
            error = np.mean((ztest - zPredict_cvd)**2)
            #error = mean_squared_error(ztest, zPredict_cvd)
            print("split: ", split+1, "Error: ", error)
            error_ols_cvd_split[split] = error
            split+=1

        #Average error of all splits per degree
        error_ols_cvd[i] = np.mean(error_ols_cvd_split)
        print("Average CVD error: ", error_ols_cvd[i])

    #Bootstrap
    if (bootstrap == True):
        X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
        X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test)
        zPredict_bootstrap = np.empty((z_test.shape[0], nBootstrap)) #Array of predicted z for each bootstrap
        for boot in range(nBootstrap):
            X_, z_ = resample(X_train, z_train) #Scikit-learn's bootstrap method
            beta_bootstrap = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
            zPredict_bootstrap[:,boot] = X_test @ beta_bootstrap #OLS prediction of the same test data for every bootstrap

        #Bootstrap results
        error_ols_bootstrap[i] = np.mean( np.mean((z_test.reshape(-1,1) - zPredict_bootstrap)**2, axis=1, keepdims=True) )
        bias_ols_bootstrap[i] = np.mean( (z_test.reshape(-1,1) - np.mean(zPredict_bootstrap, axis=1, keepdims=True))**2 )
        var_ols_bootstrap[i] = np.mean( np.var(zPredict_bootstrap, axis=1, keepdims=True) )

        print("Bootstrap")
        print("degree    error     Bias    Var")
        print("{}   {}   {}     {}".format(degrees[i], error_ols_bootstrap[i], bias_ols_bootstrap[i], var_ols_bootstrap[i]))

    if(complexity == True): #Feil (Fiks senere)
        beta_O = np.linalg.pinv(X_train_scikit.T @ X_train_scikit) @ X_train_scikit.T @ z_train_scikit
        z_tildE = X_train_scikit @ beta_O
        z_predicT = X_test_scikit @ beta_O
        err_test = np.mean((z_test_scikit - z_predicT)**2)
        bias_test = np.mean((z_test_scikit - np.mean(z_predicT))**2)
        var_test = np.mean(np.var(z_predicT))
        print(err_test, bias_test, var_test)
        e[i] = err_test
        b[i] = bias_test
        v[i] = var_test

#Plot results from bootstrap
if (bootstrap == True):
    plt.plot(degrees, error_ols_bootstrap, label="error")
    plt.plot(degrees, bias_ols_bootstrap, label="bias")
    plt.plot(degrees, var_ols_bootstrap, label="var")
    plt.legend()
    plt.show()
#Plot results from complexity
if (complexity == True):
    plt.plot(degrees, e, label='error')
    plt.plot(degrees, b, label='Bias')
    plt.plot(degrees, v, label='Var')
    plt.legend()
    plt.show()
