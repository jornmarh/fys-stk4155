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

def lassoLambda(*args):
	I = np.eye(21, 21)
	i = 0

	if (len(args) == 3):
		X, z, lambdas = args
		MSE_tilde = np.zeros(len(lambdas))
		for lmd in lambdas:
			RegLasso = linear_model.Lasso(lmd)
			RegLasso.fit(X, z)
			ztilde = RegLasso.predict(X)
			MSE_tilde[i] = MSE(z, ztilde)
			i+=1
		return MSE_tilde

	elif(len(args) == 5):
		X_train, X_test, z_train, z_test, lambdas = args
		MSE_predict = np.zeros(len(lambdas))
		MSE_tilde = np.zeros(len(lambdas))
		for lmd in lambdas:
			RegLasso = linear_model.Lasso(lmd)
			RegLasso.fit(X_train_scaled, z_train)
			ztilde = RegLasso.predict(X_train_scaled)
			zpredict = RegLasso.predict(X_test_scaled)
			MSE_tilde[i] = MSE(z_train, ztilde)
			MSE_predict[i] = MSE(z_test, zpredict)
			i+=1
		return MSE_tilde, MSE_predict

# Making meshgrid of datapoints and compute Franke's function
n = 5
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
z = FrankeFunction(xflat, yflat)
X = create_X(xflat, yflat, n=n)

	#Ridge aand Lasso nalysis of lambda parameter
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

nLambda = 500; lambdas = np.logspace(-4,8, nLambda)
mse_train_ridge, mse_test_ridge = ridgeLambda(X_train_scaled, X_test_scaled, z_train, z_test, lambdas)
mse_train_lasso, mse_test_lasso = lassoLambda(X_train_scaled, X_test_scaled, z_train, z_test, lambdas)

	#OLS (SVD)
beta_OLS = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ z_train
ztilde_OLS = X_train_scaled @ beta_OLS
zpredict_OLS = X_test_scaled @ beta_OLS
mse_OLS_train = np.zeros(len(lambdas))
mse_OLS_test = np.zeros(len(lambdas))
for i in range(len(lambdas)):
	mse_OLS_train[i] = MSE(z_train, ztilde_OLS)
	mse_OLS_test[i] = MSE(z_test, zpredict_OLS)

	#Ridge
lambda_ridge = 10**-1
_I = np.eye(21,21)

beta_ridge = np.linalg.inv(X_train_scaled.T @ X_train_scaled + lambda_ridge*_I) @ X_train_scaled.T @ z_train
ztilde_ridge = X_train_scaled @ beta_ridge
zpredict_ridge = X_test_scaled @ beta_ridge

	#Lasso
lambda_lasso = 10**-1;
RegLasso = linear_model.Lasso(lambda_lasso)
RegLasso.fit(X_train_scaled, z_train)
ztilde_lasso = RegLasso.predict(X_train_scaled)
zpredict_lasso = RegLasso.predict(X_test_scaled)


	#Compare methods by MSE Error
print('MSE error comparisson for OLS \n Train = {} \n Test = {}'.format(MSE(z_train,ztilde_OLS), MSE(z_test, zpredict_OLS)))
print('MSE error comparisson for ridge \n Train = {} \n Test = {}'.format(MSE(z_train,ztilde_ridge), MSE(z_test, zpredict_ridge)))
print('MSE error comparisson for Lasso \n Train = {} \n Test = {}'.format(MSE(z_train,ztilde_lasso), MSE(z_test, zpredict_lasso)))

fig, ax = plt.subplots()
ax.set_xlabel('log10(lambda)')
ax.set_ylabel('MSE')
ax.plot(np.log10(lambdas), mse_train_ridge, alpha=0.7,lw=2, label='Ridge train')
ax.plot(np.log10(lambdas), mse_test_ridge, alpha=0.7, lw=2, c = 'm', label = 'Ridge test')
ax.plot(np.log10(lambdas), mse_train_lasso, alpha=0.7, lw=2, c = 'g', label = 'Lasso train')
ax.plot(np.log10(lambdas), mse_test_lasso, alpha=0.7, lw=2, c = 'r', label = 'Lasso test')
ax.plot(np.log10(lambdas), mse_OLS_train, c = 'b', label = 'OLS')
ax.legend()
ax.set_title('MSE error')
plt.show()
