import numpy as np
import matplotlib.pyplot as plt
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from module1 import Sdg, Franke
from sklearn.linear_model import SGDRegressor
import seaborn as sns
import pandas as pd
sns.set()


def ols_regression(X_train, y_train):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    ytilde= X_train @ beta
    mse = mean_squared_error(y_train, ytilde)
    return mse

def ridge_regression(X_train, y_train, lmd):
     I_ = np.eye(X_train.shape[1], X_train.shape[1])
     beta = np.linalg.pinv(X_train.T @ X_train + lmd*I_) @ X_train.T @ y_train
     ytilde = X_train @ beta
     mse = mean_squared_error(y_train, ytilde)
     return mse

# Initialise data
np.random.seed(64)
N = 50
polydegree = 5
noise_coef = 0.15
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

# OLS regression for comparison
print("Analytical result from OLS:", ols_regression(X_train,z_train))
"""
# Calculate MSE as function of number of epochs
epoch_nums = np.arange(100)
batch_size = 5
mse_epochs = np.zeros(len(epoch_nums))
eta = 0.001

# MSE as function of epochs with constant learning rate
print('MSE as function of epochs with constant learning rate')
for i in range(len(epoch_nums)):
    sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, batch_size, epoch_nums[i])
    mse_epochs[i] = sgdreg.stocastichGD_ols()
    print(mse_epochs[i])

plt.plot(epoch_nums, mse_epochs, label='Batch size = %i' %(batch_size))
plt.title('MSE as function of the number of epochs')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.ylim(0,1)
plt.legend()
plt.show()

# MSE as function of epochs with different learning rates
print('MSE as function of epochs with different learning rates')
etas = [0.0001,0.0005,0.001,0.005]
mse = np.zeros((len(epoch_nums),len(etas)))
print(mse.shape)

for i in range(len(epoch_nums)):
    for j in range(len(etas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[j], batch_size, epoch_nums[i])
        mse[i,j] = sgdreg.stocastichGD_ols()
        print(mse[i,j])

plt.plot(epoch_nums, mse[:,0], label='$\eta$ = %.4f' %(etas[0]))
plt.plot(epoch_nums, mse[:,1], label='$\eta$ = %.4f' %(etas[1]))
plt.plot(epoch_nums, mse[:,2], label='$\eta$ = %.3f' %(etas[2]))
plt.plot(epoch_nums, mse[:,3], label='$\eta$ = %.3f' %(etas[3]))
plt.title('MSE as function of the number of epochs with different learning rates')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.ylim(0,1)
plt.legend()
plt.show()

"""
# MSE as function of batch size with constant learning rate
batch_sizes = np.arange(1,20)
etas = [0.0001,0.0005,0.001]
mse = np.zeros((len(batch_sizes), len(etas)))
best_mse = 1
for i in range(len(batch_sizes)):
    for j in range(len(etas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[j], int(batch_sizes[i]), 100)
        mse[i,j] = sgdreg.stocastichGD_ols()
        print(batch_sizes[i])
        if mse[i,j] < best_mse:
            best_mse = mse[i,j]
print('Lowest MSE: ', best_mse)

mse_dataframe = pd.DataFrame(mse, index = batch_sizes, columns = etas)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis_r", fmt='.4f')
#ax.set_title("Grid search for optimal $\eta$ and $\gamma$")
ax.set_xlabel("batch size")
ax.set_ylabel("learning rate")
plt.show()

"""
plt.plot(batch_sizes, mse)
plt.title("Finding the optimal batch size")
plt.xlabel("batch size")
plt.ylabel("MSE")
plt.ylim(0,1)
plt.legend()
plt.show()

# MSE as function of bach size with constant learning rate and epochs
batch_sizes = [1,2,5,10,20,50]
mse_regular = np.zeros(len(batch_sizes))
best_mse = 1
for i in range(len(batch_sizes)):
    sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.001, 5, 100)
    mse_regular[i] = sgdreg.stocastichGD_ols()
    print(etas[i])
    if mse_regular[i] < best_mse:
        best_mse = mse_regular[i]
print('Lowest MSE: ', best_mse)

plt.plot(np.log10(etas), mse_regular, label=' Epochs: %i\n Batch size: %i' %(sgdreg.n_epochs, sgdreg.M))
plt.title("Finding the optimal value for $\eta$")
plt.xlabel("$log10(\eta)$")
plt.ylabel("MSE")
plt.ylim(0,1)
plt.legend()
plt.show()

# SGD with learning learning schedule
eta = 0.0025
sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, 5, 2000)
print(sgdreg.stocastichGD_ols())


#SGD with momentum
eta = 0.006
epochs = 100
M = 5

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, 5, 100)
print(sgdreg.stocastichGD_ols('momentum', gamma=0.5))


gamma = np.arange(0,0.9, 0.01) # momentum parameter, value between 0 and 1
mse_momentum = np.zeros(len(gamma))

for i in range(len(gamma)):
    sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, 5, 100)
    mse_momentum[i] = sgdreg.stocastichGD_ols('momentum', gamma=gamma[i])
    if mse_regular[i] < best_mse:
        best_mse = mse_regular[i]
print('Lowest MSE momentum: ', best_mse)

plt.plot(gamma,mse_momentum)
plt.xlabel('$\gamma$')
plt.ylabel('MSE')
plt.show()

# RMS-prop
eta = 0.01
epochs = 100
M = 5

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
print(sgdreg.stocastichGD_ols('rmsprop', beta=0.9))

# regular SGD ridge gridsearch
epochs = 100
M = 5
lambdas = np.logspace(-8,0,9)
etas = np.logspace(-5,-2.3,5)
mse_gridsearch = np.zeros((len(lambdas), len(etas)))

sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.001, M, epochs)
sgdreg.stocastichGD_ridge(0)


for i in range(len(lambdas)):
    for j in range(len(etas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[j], M, epochs)
        mse_gridsearch[i,j] = sgdreg.stocastichGD_ridge(lambdas[i])

# Plot gridsearch
mse_dataframe = pd.DataFrame(mse_gridsearch, index = lambdas, columns = etas)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis_r", fmt='.4f')
ax.set_title("Grid search for optimal $\eta$ and $\gamma$")
ax.set_xlabel("$\eta$")
ax.set_ylabel("$\lambda$")
plt.show()
"""
