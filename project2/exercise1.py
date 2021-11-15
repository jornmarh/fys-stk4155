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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
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
N = 20
polydegree = 5
noise_coef = 0.15
x = np.sort(np.random.uniform(0, 1, N)); y = np.sort(np.random.uniform(0, 1, N))
x_mesh, y_mesh = np.meshgrid(x,y); x_flat = np.ravel(x_mesh); y_flat = np.ravel(y_mesh)

input = Franke(x_flat, y_flat, polydegree, noise_coef)
X_train, X_test, z_train, z_test = input.format()

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



"""
# MSE as function of epochs with different learning rates
print('MSE as function of epochs with different learning rates')
etas = [0.0001,0.0005,0.001,0.005]
#etas = [0.001,0.005]
epoch_nums = np.arange(1,100,2)
batch_size = 5
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

# For testing single values
#sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.001, 5, 100)
#print(sgdreg.stocastichGD_ols())


"""
# gridsearch for batch size vs learning rate
print('gridsearch for batch size vs learning rate')
batch_sizes = np.arange(5,30)
etas = [0.0001,0.0005,0.001, 0.005]
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
ax.set_title("Grid search for batch size and learning rate")
ax.set_xlabel("learning rate")
ax.set_ylabel("batch_size")
plt.show()



# SGD with learning learning schedule
print('SGD with learning learning schedule')
eta = 0.01
epochs = 10000
M = 5
sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
print(sgdreg.stocastichGD_ols(schedule=True))



#SGD with momentum
print('SGD with momentum')
eta = 0.006
epochs = 100
M = 5

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
print(sgdreg.stocastichGD_ols('momentum', gamma=0.5, schedule=True))


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
print('SGD with RMS-prop')
eta = 0.1
epochs = 100
M = 5

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
print(sgdreg.stocastichGD_ols('rmsprop', beta=0.9))

# regular SGD ridge gridsearch
print('Regular SGD ridge gridsearch')
epochs = 100
M = 5
lambdas = np.logspace(-8,0,9)
etas = np.around(np.logspace(-5,-2.3,9),decimals=5)
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
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis_r", fmt='.2f')
ax.set_title("Grid search for optimal $\eta$ and $\gamma$")
ax.set_xlabel("$\eta$")
ax.set_ylabel("$\lambda$")
plt.show()

# SGD ridge learning schedule
sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.0023, 5, 100)
sgdreg.stocastichGD_ridge(1e-8, schedule=True)

# SGD ridge momentum
sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.0023, 5, 100)
sgdreg.stocastichGD_ridge(1e-8, 'momentum',gamma=0.5)


print('MSE as function of epochs with different algorithms')
epoch_nums = np.arange(100)
batch_size = 5
mse_normal = np.zeros(len(epoch_nums))
mse_schedule = np.zeros(len(epoch_nums))
mse_momentum = np.zeros(len(epoch_nums))
mse_rmsprop = np.zeros(len(epoch_nums))

for i in range(len(epoch_nums)):
    sgdreg_normal = Sdg(X_train, X_test, z_train, z_test, 0.001, batch_size, epoch_nums[i])
    sgdreg_schedule = Sdg(X_train, X_test, z_train, z_test, 0.01, batch_size, epoch_nums[i])
    sgdreg_momentum = Sdg(X_train, X_test, z_train, z_test, 0.0015, batch_size, epoch_nums[i])
    sgdreg_rmsprop = Sdg(X_train, X_test, z_train, z_test, 0.01, batch_size, epoch_nums[i])
    mse_normal[i] = sgdreg_normal.stocastichGD_ols()
    mse_schedule[i] = sgdreg_schedule.stocastichGD_ols(schedule=True)
    mse_momentum[i] = sgdreg_momentum.stocastichGD_ols('momentum')
    mse_rmsprop[i] = sgdreg_rmsprop.stocastichGD_ols('rmsprop')

plt.plot(epoch_nums, mse_normal, label='Regular SGD')
plt.plot(epoch_nums, mse_schedule, label='With schedule')
plt.plot(epoch_nums, mse_momentum, label='With momentum')
plt.plot(epoch_nums, mse_rmsprop, label='With RMS-prop')
plt.title('MSE as function of the number of epochs with different algorithms')
plt.xlabel('Number of epochs')
plt.ylabel('MSE')
plt.ylim(0,1)
plt.legend()
plt.show()

# Scikit
sgd_scikit = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0, fit_intercept=False, max_iter=3000, \
tol=0.001, shuffle=True, random_state=68, learning_rate='constant', eta0=0.002)
sgd_scikit.fit(X_train, z_train)
z_pred_scikit = sgd_scikit.predict(X_train)
print(mean_squared_error(z_train, z_pred_scikit))

epochs = [20,50,100,200,500,1000,2000,3000]
for i in range(len(epochs)):
    sgd_scikit = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0, fit_intercept=False, max_iter=epochs[i], \
    tol=0.001, shuffle=True, random_state=68, learning_rate='constant', eta0=0.002)
    sgd_scikit.fit(X_train, z_train)
    z_pred_scikit = sgd_scikit.predict(X_train)
    print(mean_squared_error(z_train, z_pred_scikit))
"""
