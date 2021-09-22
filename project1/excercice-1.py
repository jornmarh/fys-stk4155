import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

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

def ridge(X_train, X_test, z_train, z_test, _lambda):
    _I = np.eye(21, 21)
    beta= np.linalg.pinv(X_train.T @ X_train + _lambda*_I) @ X_train.T @ z_train
    z_tilde = X_train @ beta
    z_predict = X_test @ beta
    return MSE(z_train, z_tilde), MSE(z_test, z_predict)

seed(42)
n = 5
maxdegree = 5
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
z = FrankeFunction(xflat, yflat) + 0.5*np.random.randn(N*N)

mse_tilde_list = np.zeros(maxdegree)
mse_predict_list = np.zeros(maxdegree)
for i in range(0, maxdegree):
    X = create_X(xflat, yflat, i)
    X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mse_tilde, mse_predict = ols(X_train, X_test, z_train, z_test, 1)
    mse_tilde_list[i] = mse_tilde
    mse_predict_list[i] = mse_predict

xaxis = np.arange(maxdegree)
plt.plot(xaxis, mse_tilde_list, label='train')
plt.plot(xaxis, mse_predict_list, label='test')
plt.legend()
plt.show()
