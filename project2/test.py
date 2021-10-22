import autograd.numpy as np
from autograd import elementwise_grad
from module1 import Franke
from sklearn.metrics import mean_squared_error

def MSE(beta):
    return ((X_train@beta - z_train)**2)/np.size(z_train)

#From https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
def create_miniBatches(X,y, M):
    mini_batches = []
    data = np.hstack((X, y.reshape(-1,1)))
    np.random.shuffle(data)
    m = data.shape[0] // M
    i = 0

    for i in range(m + 1):
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


np.random.seed(64)
N = 100
polydegree = 5
noise_coef = 0.15
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
ytilde = X_train @ beta
print(mean_squared_error(z_train, ytilde))

max_iter = 10000
eta = 0.05
iter = 0
beta_gd = np.random.randn(X_train.shape[1])
while iter < max_iter:
    gradient = (2.0/len(X_train))*X_train.T @ (X_train @ beta_gd - z_train)
    beta_gd = beta_gd - eta*gradient
    iter += 1
ytilde_gd = X_train @ beta_gd
print(mean_squared_error(z_train, ytilde_gd))

n_epochs = 100
eta = 0.001
M = 5
theta = np.random.randn(X_train.shape[1])
for epoch in range(n_epochs):
    mini_batches = create_miniBatches(X_train, z_train, M)
    for mini_batch in mini_batches:
        xi, yi = mini_batch
        gradient = 2.0*xi.T@((xi@theta)-yi)
        theta = theta - eta*gradient
ytilde_sdg = X_train @ theta
mse_sdg = mean_squared_error(z_train, ytilde_sdg)
print(mse_sdg)
