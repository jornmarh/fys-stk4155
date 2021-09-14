from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
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

def ridgeLambda(*args):
	I = np.eye(21, 21)
	i = 0

	if (len(args) == 3):
		X, z, lambdas = args
		MSE_tilde = np.zeros(len(lambdas))
		for lmd in lambdas:
			beta = np.linalg.inv(X.T @ X + lmd*I) @ X.T @ z
			ztilde =  X @ beta
			MSE_tilde[i] = MSE(z, ztilde)
			i+=1
		return MSE_tilde

	elif(len(args) == 5):
		X_train, X_test, z_train, z_test, lambdas = args
		MSE_predict = np.zeros(len(lambdas))
		MSE_tilde = np.zeros(len(lambdas))
		for lmd in lambdas:
			beta = np.linalg.inv(X_train.T @ X_train + lmd*I) @ X_train.T @ z_train
			ztilde =  X_train @ beta
			zPredict = X_test @ beta
			MSE_tilde[i] = MSE(z_train, ztilde)
			MSE_predict[i] = MSE(z_test, zPredict)
			i+=1
		return MSE_tilde, MSE_predict

# Making meshgrid of datapoints and compute Franke's function
n = 5
N = 1000
np.random.seed(5)
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
z = FrankeFunction(x, y)
X = create_X(x, y, n=n)

	#Ridge analysis of lambda parameter
nlambdas = 500
lambdas = np.logspace(-10, 10,nlambdas)
mse_tilde = ridgeLambda(X,z, lambdas); #print(mse_tilde)

	#Same for scaled train/test data
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mse_train, mse_test = ridgeLambda(X_train_scaled, X_test_scaled, z_train, z_test, lambdas)

fig, ax = plt.subplots(2,1, constrained_layout=True)
ax[0].set_xlabel('log10(lambda)')
ax[0].set_ylabel('MSE')
ax[0].plot(np.log10(lambdas), mse_train, alpha=0.7,lw=2, label='train')
ax[0].plot(np.log10(lambdas), mse_test, alpha=0.7, lw=2, c = 'm', label = 'test')
ax[0].legend()
ax[0].set_title('Scaled train/test data')
ax[1].plot(np.log10(lambdas), mse_tilde, alpha=0.7, lw=2, c = 'b', label = 'X')
ax[1].set_xlabel('log10(lambda)')
ax[1].set_ylabel('MSE')
ax[1].legend()
ax[1].set_title('Original data')
fig.suptitle('MSE error as a function of lambda')
plt.show()

	#OLS (SVD)
print(z_train.shape)
beta_OLS = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train
ztilde_OLS = X_train_scaled @ beta_OLS
zpredict_OLS = X_test_scaled @ beta_OLS


	#Ridge
_lambda = 10**-1; _I = np.eye(21,21)
beta_ridge = np.linalg.inv(X_train_scaled.T @ X_train_scaled + _lambda*_I) @ X_train_scaled.T @ z_train
ztilde_ridge = X_train_scaled @ beta_ridge
zpredict_ridge = X_test_scaled @ beta_ridge

	#Compare methods by MSE Error
print('MSE error comparisson for SVD OLS \n Train = {} \n Test = {}'.format(MSE(z_train,ztilde_OLS), MSE(z_test, zpredict_OLS)))
print('MSE error comparisson for ridge \n Train = {} \n Test = {}'.format(MSE(z_train,ztilde_ridge), MSE(z_test, zpredict_ridge)))
