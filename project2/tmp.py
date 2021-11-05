import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


#Initilize data
np.random.seed(64)
N = 200
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh)
yflat = np.ravel(ymesh)
print(len(xflat))
print(len(yflat))


z = (FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N))
X = np.hstack((xflat.reshape(-1,1), yflat.reshape(-1,1)))
print(X.shape)
print(z.shape)

X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

print(X_train)
print(X_train.shape)
print(z_train.shape)
