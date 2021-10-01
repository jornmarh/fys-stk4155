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
maxdegree = 15
N = 25
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
z = FrankeFunction(xflat, yflat) + 0.4*np.random.randn(N*N)

#Test complexity
m = np.zeros(maxdegree)
mseTrain = np.zeros(maxdegree)
b = np.zeros(maxdegree)
v = np.zeros(maxdegree)
degree = np.zeros(maxdegree)
mse_degree = np.zeros(maxdegree)
bias_degree = np.zeros(maxdegree)
var_degree = np.zeros(maxdegree)
nBootstrap = 100
for i in range(maxdegree):
    degree[i] = i
    X = create_X(xflat, yflat, n=i)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
    #Scale own
    X_train_own = (X_train - np.mean(X_train))/np.std(X_train)
    X_test_own = (X_test - np.mean(X_test))/np.std(X_test)
    z_train_own = (z_train - np.mean(z_train))/np.std(z_train)
    z_test_own = (z_test - np.mean(z_test))/np.std(z_test)
    #scale scikit
    scaler = StandardScaler()
    scaler_x = scaler.fit(X_train)
    X_train_scikit = scaler_x.transform(X_train)
    X_test_scikit = scaler_x.transform(X_test)
    scaler_z = scaler.fit(z_train.reshape(-1,1))
    z_train_scikit = scaler_z.transform(z_train.reshape(-1,1))
    z_test_scikit = scaler_z.transform(z_test.reshape(-1,1))
    z_train_scikit = np.ravel(z_train_scikit)
    z_test_scikit = np.ravel(z_test_scikit)

    #Bootstrap
    print("Polynomial degree    MSE     Bias    Var     Bias+Var+noise")
    mse_bootstap = np.zeros(nBootstrap)
    bias_bootstrap = np.zeros(nBootstrap)
    var_bootstrap = np.zeros(nBootstrap)
    for boot in range(nBootstrap):
        X_, z_ = resample(X_train_scikit, z_train_scikit)
        beta_ols = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
        zPredict_ols = X_test_scikit @ beta_ols
        mse_bootstap[boot] = mean_squared_error(z_test_scikit, zPredict_ols)
        bias_bootstrap[boot] = np.mean((z_test_scikit - np.mean(zPredict_ols))**2)
        var_bootstrap[boot] = np.mean(np.var(zPredict_ols))
    mse_degree[i] = np.mean(mse_bootstap)
    bias_degree[i] = np.mean(bias_bootstrap)
    var_degree[i] = np.mean(var_bootstrap)
    print("{}   {}   {}     {}     {}".format(degree[i], mse_degree[i], bias_degree[i], var_degree[i], bias_degree[i]+var_degree[i]))
plt.plot(degree, mse_degree, label="mse")
plt.plot(degree, bias_degree, label="bias")
plt.plot(degree, var_degree, label="var")
plt.legend()
plt.show()

#complexity
'''
    beta_O = np.linalg.pinv(X_train_scikit.T @ X_train_scikit) @ X_train_scikit.T @ z_train_scikit
    z_tildE = X_train_scikit @ beta_O
    z_predicT = X_test_scikit @ beta_O
    mse_test = mean_squared_error(z_test_scikit, z_predicT)
    bias_test = np.mean((z_test_scikit - np.mean(z_predicT))**2)
    var_test = np.mean(np.var(z_predicT))
    print(mse_test, bias_test, var_test)
    m[i] = mse_test
    b[i] = bias_test
    v[i] = var_test
plt.plot(degree, m, label='MSE')
plt.plot(degree, b, label='Bias')
plt.plot(degree, v, label='Var')
plt.legend()
plt.show()
'''

#CVD
'''
X = create_X(xflat, yflat, n=n)
k = 5
kfold = KFold(n_splits=k)

meanErrors = np.zeros(k)
i = 0
print("Mean squared error per kfold: ")
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

    meanerr = mean_squared_error(ztest, zPredict)
    print(meanerr)
    meanErrors[i] = meanerr
    i+=1

print("Average mse: {}".format(np.mean(meanErrors)))

plt.plot(degree, bias_degree, label='bias')
plt.plot(degree, var_degree, label='var')
plt.plot(degree, mse_degree, label='MSE')
plt.xlabel('Degree')
plt.ylabel('error')
plt.legend()
plt.show()
'''
