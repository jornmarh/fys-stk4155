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

# Function to find optimal number of epochs and minibatch size
def gridsearch_OLS(epoch_list, batch_list, eta, algo):
    mse_gridsearch = np.zeros((len(epoch_list), len(batch_list)))
    best_mse = 10
    opt_epochs = 0
    opt_batch_size = 0
    for i in range(len(epoch_list)):
        for j in range(len(batch_list)):
            sgdRegressor = Sdg(X_train, X_test, z_train, z_test, eta, batch_list[j], epoch_list[i])
            mse_gridsearch[i,j] = sgdRegressor.stocastichGD_ols(algo)

            # Store optimal number of epochs and mini-batch size
            if mse_gridsearch[i,j] < best_mse:
                best_mse = mse_gridsearch[i,j]
                opt_epochs = sgdRegressor.n_epochs
                opt_batch_size = sgdRegressor.M
    return mse_gridsearch, opt_epochs, opt_batch_size

# Plot gridsearch
epoch_list = [10,20,50,100,200]
batch_list = [5,10,15,20,30]
mse_gridsearch, opt_epochs, opt_batch_size = gridsearch_OLS(epoch_list, batch_list, 0.001, 'normalsgd')
print(mse_gridsearch)
print(" Optimal epochs: %i \n Optimal bach-size: %i"%(opt_epochs, opt_batch_size))
mse_dataframe = pd.DataFrame(mse_gridsearch, index = epoch_list, columns = batch_list)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis")
ax.set_title("Grid search for optimal batch size and number of epochs")
ax.set_ylabel("Epochs")
ax.set_xlabel("Batch size")
plt.show()


# Calculate MSE as function of number of epochs
epoch_nums = np.arange(200)
batch_size = 5
mse_epochs = np.zeros(len(epoch_nums))
eta = 0.001

for i in range(len(epoch_nums)):
    sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, batch_size, epoch_nums[i])
    mse_epochs[i] = sgdreg.stocastichGD_ols()

plt.plot(epoch_nums, mse_epochs, label='Batch size = %i' %(batch_size))
plt.title('MSE as function of the number of epochs')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.ylim(0,1)
plt.legend()
plt.show()

# Calculate MSE as function of learning rate
etas = np.logspace(-5,-2.8,50)
mse_regular = np.zeros(len(etas))
best_mse = 1
for i in range(len(etas)):
    sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[i], 5, 100)
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



#OLS with momentum
eta = 0.001
epochs = 100
M = 5

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
lambdas = np.logspace(-5,0,5)
etas = np.logspace(-5,-2.5,5)
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
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis")
ax.set_title("Grid search for optimal $\eta$ and $\gamma$")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
plt.show()
