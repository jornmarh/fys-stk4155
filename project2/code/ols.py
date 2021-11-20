'''
Code we used to obtain ols and ridge results of the franke-functio to compare
with our neural network code.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from module1 import Sdg, Franke
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)    # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)
    return X

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

np.random.seed(64)
N = 20
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh); yflat = np.ravel(ymesh)

z = (FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N))
#X = np.hstack((xflat.reshape(-1,1), yflat.reshape(-1,1)))
X = create_X(xflat, yflat, 5)

X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

model_ols = LinearRegression().fit(X_train, z_train)
ytilde_ols = model_ols.predict(X_test)
print("OLS errors")
print(mean_squared_error(z_test, ytilde_ols))
print(r2_score(z_test, ytilde_ols))
print("")

model_ridge = Ridge(alpha=1e-7).fit(X_train, z_train)
ytilde_ridge = model_ridge.predict(X_test)
print("Ridge errors")
print(mean_squared_error(z_test, ytilde_ridge))
print(r2_score(z_test, ytilde_ridge))
