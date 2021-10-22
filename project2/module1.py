import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class Sdg:
    def __init__(self, X_train, X_test, y_train, y_test, eta, M, n_epochs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.eta = eta
        self.M = M
        self.n_epochs = n_epochs
        self.m = int(len(self.X_train)/self.M)

    def grad_descent(self, max_iter):
        iter = 0
        beta = np.random.randn(self.X_train.shape[1])
        while iter < max_iter:
            gradient = (2.0/len(self.X_train))*self.X_train.T @ (self.X_train @ beta - self.y_train)
            beta = beta - self.eta*gradient
            iter += 1
        self.theta_gd = beta
        ytilde_gd = self.X_train @ self.theta_gd
        print(mean_squared_error(self.y_train, ytilde_gd))

    def stocastichGD_ols(self):
        theta = np.random.randn(self.X_train.shape[1])
        for epoch in range(self.n_epochs):
            for k in range(self.m):
                random_index = np.random.randint(self.m)
                xi = self.X_train[random_index:random_index+1]
                yi = self.y_train[random_index:random_index+1]
                gradient = 2.0*xi.T@((xi@theta)-yi)
                theta = theta - self.eta*gradient
        self.ytilde_sdg_ols = self.X_train @ theta
        mse_sdg_ols = mean_squared_error(self.y_train, self.ytilde_sdg_ols)
        print(mse_sdg_ols)

    def stocastichGD_ridge(self, lmd):
        theta = np.random.randn(self.X_train.shape[1])
        for epoch in range(self.n_epochs):
            for k in range(self.m):
                random_index =  np.random.randint(self.m)
                xi = self.X_train[random_index:random_index+1]
                yi = self.y_train[random_index:random_index+1]
                gradient = 2.0*xi.T@((xi@theta)-yi) + 2.0*lmd*theta
                theta = theta - self.eta*gradient
        self.ytilde_sdg_ridge = self.X_train @ theta
        mse_sdg_ridge = mean_squared_error(self.y_train, self.ytilde_sdg_ridge)
        print(mse_sdg_ridge)

class Franke:
    def __init__(self, x, y, polydegree, noise):
        self.x = x
        self.y = y
        self.degree = polydegree
        self.noise_coef = noise
        self.N = len(x)

    def FrankeFunction(self):
        term1 = 0.75*np.exp(-(0.25*(9*self.x-2)**2) - 0.25*((9*self.y-2)**2))
        term2 = 0.75*np.exp(-((9*self.x+1)**2)/49.0 - 0.1*(9*self.y+1))
        term3 = 0.5*np.exp(-(9*self.x-7)**2/4.0 - 0.25*((9*self.y-3)**2))
        term4 = -0.2*np.exp(-(9*self.x-4)**2 - (9*self.y-7)**2)
        noise = self.noise_coef * np.random.randn(self.N)
        return term1 + term2 + term3 + term4 + noise

    def create_X(self):
        n = self.degree
        l = int((n+1)*(n+2)/2) # Number of columns in beta
        X = np.ones((self.N,l))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (self.x**(i-k))*(self.y**k)

        return X

    def scale(self, X, z):
        X_train, X_test, z_train, z_test = train_test_split(X,z)
        scaler = StandardScaler() # Utilizing scikit's standardscaler
        scaler_x = scaler.fit(X_train) # Scaling x-data
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)
        scaler_z = scaler.fit(z_train.reshape(-1,1)) # Scaling z-data
        z_train = scaler_z.transform(z_train.reshape(-1,1)).ravel()
        z_test = scaler_z.transform(z_test.reshape(-1,1)).ravel()
        return X_train, X_test, z_train, z_test

    def format(self):
        z = self.FrankeFunction()
        X = self.create_X()
        return self.scale(X, z)
