import autograd.numpy as np
from autograd import elementwise_grad
from module1 import Franke
from sklearn.metrics import mean_squared_error

def MSE(beta):
    return ((X_train@beta - z_train)**2)/np.size(z_train)

def schedule(t):
    return t_0/(t + t_1)

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
'''
#Gradient descent
max_iter = 10000
eta = 0.005
iter = 0
beta_gd = np.random.randn(X_train.shape[1])
while iter < max_iter:
    gradient = (2.0/len(X_train))*X_train.T @ (X_train @ beta_gd - z_train)
    beta_gd = beta_gd - eta*gradient
    iter += 1
ytilde_gd = X_train @ beta_gd
print(mean_squared_error(z_train, ytilde_gd))
'''
'''
#My method without learning schedule
n_epochs = 100
eta = 1/1000
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
'''

#My method with schedule
itr = 0
while itr < 50:
    t_0 = 1
    t_1 = 50
    n_epochs = 100
    M = 5
    m = int(len(X_train)/M)
    theta_ls = np.random.randn(X_train.shape[1])
    for epoch in range(n_epochs):
        mini_batches = create_miniBatches(X_train, z_train, M)
        k = 0
        for mini_batch in mini_batches:
            xi, yi = mini_batch
            gradient_ls = 2.0*xi.T@((xi@theta_ls)-yi)
            eta_ls = schedule(epoch*m+k)
            theta_ls = theta_ls - eta_ls*gradient_ls
            k+=1
    ytilde_sdg_ls = X_train @ theta_ls
    mse_sdg_ls = mean_squared_error(z_train, ytilde_sdg_ls)
    print(mse_sdg_ls)
    itr += 1
'''
#Morten with learning schedule
t_0 = 1
t_1 = 50
n_epochs = 100
M = 5
m = int(len(X_train)/M)
beta = np.random.randn(X_train.shape[1])
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_train[random_index:random_index+1]
        yi = z_train[random_index:random_index+1]
        gradients = 2 * xi.T @ ((xi @ beta)-yi)
        eta = schedule(epoch*m+i)
        beta = beta - eta*gradients
ytilde_sdg_copy = X_train @ beta
mse_sdg_copy = mean_squared_error(z_train, ytilde_sdg_copy)
print(mse_sdg_copy)
'''
