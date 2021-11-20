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

        self.mse = []
        self.r2 = []
        self.epochs = []

    # Method for creating minibatches for SGD
    def create_miniBatches(self, X, y, M):
        mini_batches = []
        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)
        m = data.shape[0] // M
        i=0
        for i in range(m):
            mini_batch = data[i * M:(i + 1)*M, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % M != 0:
            mini_batch = data[i * M:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def schedule(self, decay, epoch):
        return self.eta/(1 + decay*epoch)

    def grad_descent(self, max_iter):
        np.random.seed(64)
        iter = 0
        beta = np.random.randn(self.X_train.shape[1])
        while iter < max_iter:
            g = (2.0/len(self.X_train))*self.X_train.T @ (self.X_train @ beta - self.y_train)
            beta = beta - self.eta*g
            iter += 1
        self.theta_gd = beta
        ytilde_gd = self.X_train @ self.theta_gd
        mse = mean_squared_error(self.y_train, ytilde_gd)
        return mse

    def stocastichGD_ols(self, algo='normalsgd', gamma=0.01, beta=0.9, eps=1e-8, schedule=False, decay=1e-6):
        np.random.seed(64)
        theta = np.random.randn(self.X_train.shape[1])

        # RMSprop second moment term
        s = np.random.normal(1,0.15,self.X_train.shape[1])

        # Momentum velocity term
        v = np.random.randn(self.X_train.shape[1])

        if algo == "normalsgd":
            for epoch in range(self.n_epochs):
                mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                for mini_batch in mini_batches:
                    xi,yi = mini_batch
                    g = 2.0*xi.T@((xi@theta)-yi)
                    if schedule == True:
                        self.eta = self.schedule(decay, epoch)
                    theta = theta - self.eta*g
                # Get errors after each epoch
                y_pred_epoch = self.X_test @ theta
                self.epochs.append(epoch)
                self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                self.r2.append(r2_score(self.y_test, y_pred_epoch))

            self.ytilde_sgd_ols = self.X_test @ theta
            mse_sgd_ols = mean_squared_error(self.y_test, self.ytilde_sgd_ols)
            r2_sgd_ols = r2_score(self.y_test, self.ytilde_sgd_ols)

            return mse_sgd_ols, r2_sgd_ols

        elif algo == "momentum":
            for epoch in range(self.n_epochs):
                mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                for mini_batch in mini_batches:
                    xi,yi = mini_batch
                    g = 2.0*xi.T @ ((xi @ theta)-yi)
                    v = gamma*v - self.eta*g
                    if schedule == True:
                        self.eta = self.schedule(1e-6, epoch)
                    theta = theta + v

                y_pred_epoch = self.X_test @ theta
                self.epochs.append(epoch)
                self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                self.r2.append(r2_score(self.y_test, y_pred_epoch))

            self.ytilde_sgd_ols = self.X_test @ theta
            mse_sgd_ols = mean_squared_error(self.y_test, self.ytilde_sgd_ols)
            r2_sgd_ols = r2_score(self.y_test, self.ytilde_sgd_ols)

            return mse_sgd_ols, r2_sgd_ols

        elif algo == "rmsprop":
            for epoch in range(self.n_epochs):
                mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                for mini_batch in mini_batches:
                    xi,yi = mini_batch
                    g = 2.0*xi.T @ ((xi @ theta)-yi)
                    s = beta*s + (1-beta)*np.dot(g,g)
                    d_theta = self.eta/(np.sqrt(s+eps))*g
                    theta = theta - d_theta

                y_pred_epoch = self.X_test @ theta
                self.epochs.append(epoch)
                self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                self.r2.append(r2_score(self.y_test, y_pred_epoch))

            self.ytilde_sgd_ols = self.X_test @ theta
            mse_sgd_ols = mean_squared_error(self.y_test, self.ytilde_sgd_ols)
            r2_sgd_ols = r2_score(self.y_test, self.ytilde_sgd_ols)

            return mse_sgd_ols, r2_sgd_ols



    def stocastichGD_ridge(self, lmd, algo='normalsgd', gamma=0.01, beta=0.9, eps=1e-8, schedule=False):
        np.random.seed(64)
        theta = np.random.randn(self.X_train.shape[1])

        # RMSprop second moment term
        s = np.random.normal(1,0.15,self.X_train.shape[1]) # Must be initially positive

        # SGD velocity term
        v = np.random.randn(self.X_train.shape[1])

        if algo == "normalsgd":
            for epoch in range(self.n_epochs):
                mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                for mini_batch in mini_batches:
                    xi,yi = mini_batch
                    g = 2.0*xi.T@((xi@theta)-yi) + 2.0*lmd*theta
                    if schedule == True:
                        self.eta = self.schedule(1e-6, epoch)
                    theta = theta - self.eta*g

                y_pred_epoch = self.X_test @ theta
                self.epochs.append(epoch)
                self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                self.r2.append(r2_score(self.y_test, y_pred_epoch))

            self.ytilde_sgd_ridge = self.X_test @ theta
            mse_sgd_ridge = mean_squared_error(self.y_test, self.ytilde_sgd_ridge)
            r2_sgd_ridge = r2_score(self.y_test, self.ytilde_sgd_ridge)
            self.epochs.append(epoch)
            self.mse.append(mse_sgd_ridge)
            self.r2.append(r2_sgd_ridge)

            return mse_sgd_ridge, r2_sgd_ridge

        elif algo == "momentum":
                for epoch in range(self.n_epochs):
                    mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                    for mini_batch in mini_batches:
                        xi,yi = mini_batch
                        g = 2.0*xi.T@((xi@theta)-yi) + 2.0*lmd*theta
                        v = gamma*v - self.eta*g
                        if schedule == True:
                            self.eta = self.schedule(1e-6, epoch)
                        theta = theta + v

                    y_pred_epoch = self.X_test @ theta
                    self.epochs.append(epoch)
                    self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                    self.r2.append(r2_score(self.y_test, y_pred_epoch))

                self.ytilde_sgd_ridge = self.X_test @ theta
                mse_sgd_ridge = mean_squared_error(self.y_test, self.ytilde_sgd_ridge)
                r2_sgd_ridge = r2_score(self.y_test, self.ytilde_sgd_ridge)
                self.epochs.append(epoch)
                self.mse.append(mse_sgd_ridge)
                self.r2.append(r2_sgd_ridge)

                return mse_sgd_ridge, r2_sgd_ridge

        elif algo == "rmsprop":
            for epoch in range(self.n_epochs):
                mini_batches = self.create_miniBatches(self.X_train, self.y_train, self.M)
                for mini_batch in mini_batches:
                    xi,yi = mini_batch
                    g = 2.0*xi.T@((xi@theta)-yi) + 2.0*lmd*theta
                    s = beta*s + (1-beta)*np.dot(g,g)
                    d_theta = self.eta/(np.sqrt(s+eps))*g
                    theta = theta - d_theta

                y_pred_epoch = self.X_test @ theta
                self.epochs.append(epoch)
                self.mse.append(mean_squared_error(self.y_test, y_pred_epoch))
                self.r2.append(r2_score(self.y_test, y_pred_epoch))

            self.ytilde_sgd_ridge = self.X_test @ theta
            mse_sgd_ridge = mean_squared_error(self.y_test, self.ytilde_sgd_ridge)
            r2_sgd_ridge = r2_score(self.y_test, self.ytilde_sgd_ridge)
            self.epochs.append(epoch)
            self.mse.append(mse_sgd_ridge)
            self.r2.append(r2_sgd_ridge)

            return mse_sgd_ridge, r2_sgd_ridge


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
        scaler = StandardScaler(with_mean=False) # Remove intercept before scaling 
        scaler_x = scaler.fit(X_train) # Scaling x-data
        X_train = scaler_x.transform(X_train)
        X_test = scaler_x.transform(X_test)
        return X_train, X_test, z_train, z_test

    def format(self):
        z = self.FrankeFunction()
        X = self.create_X()
        return self.scale(X, z)
