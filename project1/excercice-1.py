import matplotlib.pyplot as plt
import numpy as np
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

def create_X(x, y, n ):
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

def noiseTest():
    noiz = np.linspace(0, 0.15, 50)
    mse_train = np.zeros(len(noiz))
    mse_test = np.zeros(len(noiz))

    for i in range(len(noiz)):
        _z = FrankeFunction(xflat, yflat) + noiz[i]*np.random.randn(N*N)
        X_train, X_test, z_train, z_test = train_test_split(X,_z, test_size=0.2)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)
        mse_train[i], mse_test[i] = ols(X_train, X_test, z_train, z_test, 1)
    plt.plot(noiz, mse_train, label='train')
    plt.plot(noiz, mse_test, label='test')
    plt.legend()
    plt.xlabel('Noise')
    plt.ylabel('MSE')
    plt.show()

#Initilize data
seed(42)
n = 5
maxdegree = 12
N = 500
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
z = FrankeFunction(xflat, yflat) + 0.1*np.random.randn(N*N)
scaler = StandardScaler()

nBootstrap = 100
degree = np.zeros(maxdegree)
mse_degree = np.zeros(maxdegree)
bias_degree = np.zeros(maxdegree)
var_degree = np.zeros(maxdegree)
print("Polynomial degree    MSE     Bias    Var")
for deg in range(maxdegree):
    X = create_X(xflat, yflat, deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    scaler_x = scaler.fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)

    degree[deg] = deg+1
    mse_bootstap = np.zeros(nBootstrap)
    bias_bootstrap = np.zeros(nBootstrap)
    var_bootstrap = np.zeros(nBootstrap)
    for boot in range(nBootstrap):
        X_, z_ = resample(X_train, z_train)
        beta_ols = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
        zPredict_ols = X_test @ beta_ols
        mse_bootstap[boot] = MSE(z_test, zPredict_ols)
        bias_bootstrap[boot] = np.mean((z_test - np.mean(zPredict_ols))**2)
        var_bootstrap[boot] = np.mean(np.var(zPredict_ols))
    mse_degree[deg] = np.mean(mse_bootstap)
    bias_degree[deg] = np.mean(bias_bootstrap)
    var_degree[deg] = np.mean(var_bootstrap)
    print("{}   {}   {}  {}".format(degree[deg], mse_degree[deg], bias_degree[deg], var_degree[deg]))


X = create_X(xflat, yflat, n=n)
k = 5
kfold = KFold(n_splits=k)

for train_inds, test_inds in kfold.split(X):
    xtrain = X[train_inds]
    ztrain = z[train_inds]
    xtest = X[test_inds]
    ztest = z[test_inds]

    scaler_x = scaler.fit(xtrain)
    xtrain = scaler_x.transform(xtrain)
    xtest = scaler_x.transform(xtest)

    beta = np.linalg.pinv(xtrain.T @ xtrain) @ xtrain.T @ ztrain
    zPredict = xtest @ beta

    print(mean_squared_error(ztest, zPredict))

plt.plot(degree, bias_degree, label='bias')
plt.plot(degree, var_degree, label='var')
plt.plot(degree, mse_degree, label='MSE')
plt.xlabel('Degree')
plt.ylabel('error')
plt.legend()
plt.show()
