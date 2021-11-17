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

'''--------------------------------------------------
                # OLS REGRESSION
---------------------------------------------------'''
"""
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


# For testing single values
sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.001, 5, 1000)
print(sgdreg.stocastichGD_ols())



# Error as function of epochs using different learning rates

etas = [0.0001,0.0005,0.001, 0.0015]
epoch = 500
sgdreg1 = Sdg(X_train, X_test, z_train, z_test, etas[0], 10, epoch)
sgdreg2 = Sdg(X_train, X_test, z_train, z_test, etas[1], 10, epoch)
sgdreg3 = Sdg(X_train, X_test, z_train, z_test, etas[2], 10, epoch)
sgdreg4 = Sdg(X_train, X_test, z_train, z_test, etas[3], 10, epoch)

mse_1, r2_1 = sgdreg1.stocastichGD_ols()
mse_2, r2_2 = sgdreg2.stocastichGD_ols()
mse_3, r2_3 = sgdreg3.stocastichGD_ols()
mse_4, r2_4 = sgdreg4.stocastichGD_ols()

print("MSE and r2 score with different learning rates")
print("lr: %.4f:\n mse = %4f\n r2 = %.4f" %(etas[0], mse_1, r2_1))
print("lr: %.4f:\n mse = %4f\n r2 = %.4f" %(etas[1], mse_2, r2_2))
print("lr: %.4f:\n mse = %4f\n r2 = %.4f" %(etas[2], mse_3, r2_3))
print("lr: %.4f:\n mse = %4f\n r2 = %.4f" %(etas[3], mse_4, r2_4))

plt.plot(sgdreg1.epochs, sgdreg1.mse, label='$\eta = %.4f$'%(etas[0]))
plt.plot(sgdreg2.epochs, sgdreg2.mse, label='$\eta = %.4f$'%(etas[1]))
plt.plot(sgdreg3.epochs, sgdreg3.mse, label='$\eta = %.4f$'%(etas[2]))
plt.plot(sgdreg4.epochs, sgdreg4.mse, label='$\eta = %.4f$'%(etas[3]))
plt.legend()
plt.title('Test mse error as function of epochs')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.ylim(0,0.25)
plt.show()

plt.plot(sgdreg1.epochs, sgdreg1.r2, label='$\eta = %.4f$'%(etas[0]))
plt.plot(sgdreg2.epochs, sgdreg2.r2, label='$\eta = %.4f$'%(etas[1]))
plt.plot(sgdreg3.epochs, sgdreg3.r2, label='$\eta = %.4f$'%(etas[2]))
plt.plot(sgdreg4.epochs, sgdreg4.r2, label='$\eta = %.4f$'%(etas[3]))
plt.legend()
plt.title('Test $r^2$ error as function of epochs')
plt.xlabel('epochs')
plt.ylabel('$r^2$ error')
plt.ylim(0,0.8)
plt.show()



print("mse after 200 epochs: ", sgdreg3.mse[199])


# gridsearch for batch size vs learning rate
print('gridsearch for batch size vs learning rate')
batch_sizes = [4,6,8,10,12,14,16]
etas = [0.0001,0.0005, 0.001, 0.0015, 0.002, 0.0022]
mse = np.zeros((len(batch_sizes), len(etas)))
r2 = np.zeros((len(batch_sizes), len(etas)))
best_mse = 1
best_r2 = 0
for i in range(len(batch_sizes)):
    for j in range(len(etas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[j], int(batch_sizes[i]), 100)
        mse[i,j], r2[i,j] = sgdreg.stocastichGD_ols()
        print(batch_sizes[i])
        if mse[i,j] < best_mse:
            best_mse = mse[i,j]
        if r2[i,j] > best_r2:
            best_r2 = r2[i,j]
print('Lowest MSE: ', best_mse)
print('Highest $r^2$: ', best_r2)

mse_dataframe = pd.DataFrame(mse, index = batch_sizes, columns = etas)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis_r", fmt='.4f')
ax.set_title("Grid search for batch size and learning rate")
ax.set_xlabel("learning rate")
ax.set_ylabel("batch_size")
plt.show()

r2_dataframe = pd.DataFrame(r2, index = batch_sizes, columns = etas)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(r2_dataframe, annot=True, ax=ax, cmap="viridis", fmt='.4f')
ax.set_title("Grid search for batch size and learning rate")
ax.set_xlabel("learning rate")
ax.set_ylabel("batch_size")
plt.show()


# SGD with learning learning schedule
print('SGD with learning learning schedule')
eta = 0.0035
epochs = 500
M = 10
decay = 1e-6
sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
mse, r2 = sgdreg.stocastichGD_ols(schedule=True, decay=decay)
print("MSE: ", mse)
print("r2 score: ", r2)
plt.plot(sgdreg.epochs,sgdreg.mse, label=" decay = %.2e \n $\eta = %.3f$" %(decay, eta))
plt.xlabel("epochs")
plt.ylabel("mse")
plt.legend()
plt.ylim(0,0.25)
plt.title("MSE as function of epochs using SGD with learning schedule")
plt.show()

plt.plot(sgdreg.epochs,sgdreg.r2, label=" decay = %.2e \n $\eta = %.3f$" %(decay, eta))
plt.xlabel("epochs")
plt.ylabel("$r^2$ error")
plt.legend()
plt.ylim(0,0.8)
plt.title("$r^2$ as function of epochs using SGD with learning schedule")
plt.show()


#SGD with momentum
print('SGD with momentum')
eta = 0.003
epochs = 500
gamma = 0.6

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, 10, epochs)
mse, r2 = sgdreg.stocastichGD_ols('momentum', gamma=gamma, schedule=True)

plt.plot(sgdreg.epochs, sgdreg.mse, label=" $\gamma = %.1f$\n $\eta = %.3f$" %(gamma, eta))
plt.legend()
plt.ylim(0,0.25)
plt.title("MSE as function of epochs with SGD momentum")
plt.show()

plt.plot(sgdreg.epochs, sgdreg.r2, label=" $\gamma = %.1f$\n $\eta = %.3f$" %(gamma, eta))
plt.legend()
plt.ylim(0,0.8)
plt.title("$r^2$ as function of epochs with SGD momentum")
plt.show()

print("Errors with SGD momentum")
print("MSE: ", mse)
print("$r^2$: ", r2)


# RMS-prop
print('SGD with RMS-prop')

eta = 0.01
epochs = 500
M = 10
beta = 0.5
decay = 1e-6

sgdreg = Sdg(X_train, X_test, z_train, z_test, eta, M, epochs)
mse, r2 = sgdreg.stocastichGD_ols('rmsprop', beta=beta, schedule=True, decay=decay)
print(mse,r2)

plt.plot(sgdreg.epochs, sgdreg.mse, label=" beta = %.1f\n $\eta = %.3f$" %(beta, eta))
plt.legend()
plt.ylim(0,0.25)
plt.title("MSE as function of epochs with RMS-prop")
plt.show()

plt.plot(sgdreg.epochs, sgdreg.r2, label=" beta = %.1f\n $\eta = %.3f$" %(beta, eta))
plt.legend()
plt.ylim(0,0.8)
plt.title("$r^2$ as function of epochs with RMS-prop")
plt.show()

print("Errors with RMS-prop")
print("MSE: ", mse)
print("$r^2$: ", r2)

# Gridsearch for eta and gamma
etas = [0.005,0.01,0.015,0.02,0.025, 0.03]
betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
mse = np.zeros((len(etas), len(betas)))
r2 = np.zeros((len(etas), len(betas)))
for i in range(len(etas)):
    for j in range(len(betas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[i], 10, 500)
        mse[i,j], r2[i,j] = sgdreg.stocastichGD_ols(algo="rmsprop", beta=betas[j])
        print(mse[i,j])

mse_df_test = pd.DataFrame(mse, index = etas, columns = betas)
fig, ax = plt.subplots(figsize = (9, 9))
sns.heatmap(mse_df_test, annot=True, ax=ax, cmap="viridis_r", fmt='.3f')
ax.set_title("Gridsearch showing mse")
ax.set_xlabel("$beta$")
ax.set_ylabel("$\eta$")
plt.show()

mse_df_test = pd.DataFrame(r2, index = etas, columns = betas)
fig, ax = plt.subplots(figsize = (9, 9))
sns.heatmap(mse_df_test, annot=True, ax=ax, cmap="viridis", fmt='.3f')
ax.set_title("Gridsearch showing $r^2$ error")
ax.set_xlabel("$beta$")
ax.set_ylabel("$\eta$")
plt.show()

'''--------------------------------------------------
                # RIDGE REGRESSION
---------------------------------------------------'''


# regular SGD ridge gridsearch
print('Regular SGD ridge gridsearch')
epochs = 500
M = 10
lambdas = np.logspace(-8,0,9)
etas = [0.0001,0.0005,0.001,0.0015,0.002,0.0025,0.0030]
mse_gridsearch = np.zeros((len(etas), len(lambdas)))
r2_gridsearch = np.zeros((len(etas), len(lambdas)))
#sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.001, M, epochs)
#sgdreg.stocastichGD_ridge(0)

for i in range(len(etas)):
    for j in range(len(lambdas)):
        sgdreg = Sdg(X_train, X_test, z_train, z_test, etas[i], M, epochs)
        mse_gridsearch[i,j], r2_gridsearch[i,j] = sgdreg.stocastichGD_ridge(lambdas[j])

# Plot gridsearch
mse_dataframe = pd.DataFrame(mse_gridsearch, index = etas, columns = lambdas)
fig, ax = plt.subplots(figsize = (9, 9))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis_r", fmt='.3f')
ax.set_title("Test mse for gridsearch of $\eta$ and $\lambda$")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
plt.show()

mse_dataframe = pd.DataFrame(r2_gridsearch, index = etas, columns = lambdas)
fig, ax = plt.subplots(figsize = (9, 9))
sns.heatmap(mse_dataframe, annot=True, ax=ax, cmap="viridis", fmt='.3f')
ax.set_title("Test $r^2$ error for gridsearch of $\eta$ and $\lambda$")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
plt.show()


# SGD ridge learning schedule
sgdreg = Sdg(X_train, X_test, z_train, z_test, 0.002, 10, 500)
print(sgdreg.stocastichGD_ridge(1e-5, schedule=False))

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
sgd_scikit = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0, fit_intercept=False, max_iter=200, \
tol=0.001, shuffle=True, random_state=68, learning_rate='constant', eta0=0.001)
sgd_scikit.fit(X_train, z_train)
z_pred_scikit = sgd_scikit.predict(X_test)
print("Scikit SGDRegressor errors")
print(mean_squared_error(z_test, z_pred_scikit))
print(r2_score(z_test, z_pred_scikit))


epochs = [20,50,100,200,500,1000,2000,3000]
for i in range(len(epochs)):
    sgd_scikit = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0, fit_intercept=False, max_iter=epochs[i], \
    tol=0.001, shuffle=True, random_state=68, learning_rate='constant', eta0=0.002)
    sgd_scikit.fit(X_train, z_train)
    z_pred_scikit = sgd_scikit.predict(X_train)
    print(mean_squared_error(z_train, z_pred_scikit))
"""
