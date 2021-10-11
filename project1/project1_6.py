import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold
from imageio import imread


'''
In this script, cross-validaton, bootstrap, and ordinary complexity analysis
is done by setting the value of bootstap, cvd, poly = True
'''


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)        # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)
    return X


def scale(X_train, X_test, z_train, z_test):
    scaler = StandardScaler()  # Utilizing scikit's standardscaler

    scaler_x = scaler.fit(X_train)  # Scaling x-data
    X_train_scikit = scaler_x.transform(X_train)
    X_test_scikit = scaler_x.transform(X_test)

    scaler_z = scaler.fit(z_train.reshape(-1, 1))  # Scaling z-data
    z_train_scikit = scaler_z.transform(z_train.reshape(-1, 1)).ravel()
    z_test_scikit = scaler_z.transform(z_test.reshape(-1, 1)).ravel()

    return X_train_scikit, X_test_scikit, z_train_scikit, z_test_scikit


# Initilize data
terrain1 = imread("SRTM_data_Norway_1.tif")

N = 25
maxdegree = 10

z = terrain1[::N, ::N] #Pick every Nth(25) point in terrain1

x = np.linspace(0, 1, len(z[0]))
y = np.linspace(0, 1, len(z[:,0]))

x_mesh, y_mesh = np.meshgrid(x, y)

xflat = x_mesh.ravel()
yflat = y_mesh.ravel()

z = z.ravel()
scaler = StandardScaler()
zScale = scaler.fit(z.reshape(-1,1))
z = zScale.transform(z.reshape(-1,1)).ravel()


X = create_X(xflat, yflat, 6)
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)
X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test)

beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
zTilde = X_train @ beta
_str = len(beta)
variance = np.zeros(_str)
CI = np.zeros(_str)

for i in range(_str):
    variance[i] = np.sqrt(mean_squared_error(z_train, zTilde))**2 * np.linalg.pinv(X_train.T @ X_train)[i,i]
    CI[i] = 2*np.sqrt(variance[i])/np.sqrt(N)
print("")
plt.errorbar(np.arange(0,_str), beta, yerr=CI, fmt='o', ecolor='red', capsize=5, markersize=4, label='beta_i with confidence range')
plt.xlabel('i')
plt.ylabel('beta [i]')
plt.legend()
plt.title("Confidence interval of the paramateres beta for a 6th order polynomial fit")
plt.show()

# complexity
poly = False
if (poly == True):
    mse_ols = np.zeros(maxdegree)
    mse_ols_train = np.zeros(maxdegree)
    r2_train = np.zeros(maxdegree)
    r2_test = np.zeros(maxdegree)
    mse_ridge = np.zeros(maxdegree)
    mse_lasso = np.zeros(maxdegree)

# Bootstrap
# Change to True to perform bootstrap analysis
bootstrap = False
nBootstrap = 1000  # Number of bootstraps
if (bootstrap == True):
    error_ols_bootstrap = np.zeros(maxdegree)
    bias_ols_bootstrap = np.zeros(maxdegree)
    var_ols_bootstrap = np.zeros(maxdegree)

    error_ridge_bootstrap = np.zeros(maxdegree)
    bias_ridge_bootstrap = np.zeros(maxdegree)
    var_ridge_bootstrap = np.zeros(maxdegree)

    error_lasso_bootstrap = np.zeros(maxdegree)
    bias_lasso_bootstrap = np.zeros(maxdegree)
    var_lasso_bootstrap = np.zeros(maxdegree)

# Cross-validation
# Change to True to perform cross validation
cvd = False
if (cvd == True):
    k = 5
    kfold = KFold(n_splits=k, shuffle=True)
    error_ols_cvd = np.zeros(maxdegree)
    error_ridge_cvd = np.zeros(maxdegree)
    error_lasso_cvd = np.zeros(maxdegree)

# Hyperparameter
lmd = 1e-3

complexity = True
# Complexity analysis

if (complexity == True):
    print("     Lambda = ", lmd, "\n")

    degrees = np.zeros(maxdegree)
    for i in range(maxdegree):
        # Create design matrix for every degree until maxdegree
        degrees[i] = i
        X = create_X(xflat, yflat, i)

        print("\n", "Degree: ", degrees[i])
        # Perform cvd analysis

        if (poly == True):
            X_train, X_test, z_train, z_test = train_test_split(X,z, test_size = 0.2)
            X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test)

            # OLS prediction
            beta_ols = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
            zTilde_ols = X_train @ beta_ols
            zPredict_ols = X_test @ beta_ols

            # ridge
            _I = np.eye(X_train.shape[1], X_train.shape[1])
            beta_ridge = np.linalg.pinv(X_train.T @ X_train + lmd*_I) @ X_train.T @ z_train
            zPredict_ridge = X_test @ beta_ridge

            # Lasso
            RegLasso = linear_model.Lasso(lmd, max_iter=1e5, tol = 0.1)
            RegLasso.fit(X_train, z_train)
            zPredict_lasso = RegLasso.predict(X_test)

            mse_ols_train[i] = mean_squared_error(z_train, zTilde_ols)
            mse_ols[i] = mean_squared_error(z_test, zPredict_ols)
            r2_train[i] = r2_score(z_train, zTilde_ols)
            r2_test[i] = r2_score(z_test, zPredict_ols)

            mse_ridge[i] = mean_squared_error(z_test, zPredict_ridge)
            mse_lasso[i] = mean_squared_error(z_test, zPredict_lasso)

            print(mse_ols[i], mse_ridge[i], mse_lasso[i])

        if(cvd == True):
            split = 0  # Variable to keep track of the specific split

            error_ols_cvd_split = np.zeros(k)
            error_ridge_cvd_split = np.zeros(k)
            error_lasso_cvd_split = np.zeros(k)
            print("CVD")
            for train_inds, test_inds in kfold.split(X):
                # Split and scale data
                xtrain = X[train_inds]
                ztrain = z[train_inds]
                xtest = X[test_inds]
                ztest = z[test_inds]
                xtrain, xtest, ztrain, ztest = scale(xtrain, xtest, ztrain, ztest)

                # OLS prediction
                beta_ols_cvd = np.linalg.pinv(xtrain.T @ xtrain) @ xtrain.T @ ztrain
                zPredict_ols_cvd = xtest @ beta_ols_cvd

                # ridge
                _I = np.eye(xtrain.shape[1], xtrain.shape[1])
                beta_ridge_cvd = np.linalg.pinv(xtrain.T @ xtrain + lmd*_I) @xtrain.T @ ztrain
                zPredict_ridge_cvd = xtest @ beta_ridge_cvd

                # Lasso
                RegLasso = linear_model.Lasso(lmd, max_iter=1e5)
                RegLasso.fit(xtrain, ztrain)
                zPredict_lasso_cvd = RegLasso.predict(xtest)


                error_ols = np.mean((ztest - zPredict_ols_cvd)**2)
                error_ridge = np.mean((ztest - zPredict_ridge_cvd)**2)
                error_lasso = np.mean((ztest - zPredict_lasso_cvd)**2)

                print("split: ", split+1, "Error OLS : ", error_ols, "    Error ridge: ", error_ridge, "     Error lasso: ", error_lasso)

                error_ols_cvd_split[split] = error_ols
                error_ridge_cvd_split[split] = error_ridge
                error_lasso_cvd_split[split] = error_lasso
                split += 1

            # Average error of all splits per degree
            error_ols_cvd[i] = np.mean(error_ols_cvd_split)
            error_ridge_cvd[i] = np.mean(error_ridge_cvd_split)
            error_lasso_cvd[i] = np.mean(error_lasso_cvd_split)

            print("Average CVD OLS error: ", error_ols_cvd[i])
            print("Average CVD Ridge error: ", error_ridge_cvd[i])
            print("Average CVD Lasso error: ", error_lasso_cvd[i])

        # Bootstrap
        if (bootstrap == True):
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            X_train, X_test, z_train, z_test = scale(X_train, X_test, z_train, z_test)

            # Array of predicted z for each bootstrap
            zPredict_ols_bootstrap = np.empty((z_test.shape[0], nBootstrap))
            zPredict_ridge_bootstrap = np.empty((z_test.shape[0], nBootstrap))
            zPredict_lasso_bootstrap = np.empty((z_test.shape[0], nBootstrap))

            for boot in range(nBootstrap):
                # Scikit-learn's bootstrap method
                X_, z_ = resample(X_train, z_train)

                # OLS
                beta_ols_bootstrap = np.linalg.pinv(X_.T @ X_) @ X_.T @ z_
                zPredict_ols_bootstrap[:, boot] = X_test @ beta_ols_bootstrap

                # Ridge
                _I = np.eye(X_train.shape[1], X_train.shape[1])
                beta_ridge_bootstrap = np.linalg.pinv(X_.T @ X_ + lmd*_I) @ X_.T @ z_
                zPredict_ridge_bootstrap[:,boot] = X_test @ beta_ridge_bootstrap

                # Lasso
                RegLasso = linear_model.Lasso(lmd, max_iter=1e5)
                RegLasso.fit(X_train, z_train)
                zPredict_lasso_bootstrap[:, boot] = RegLasso.predict(X_test)


            # Bootstrap results

            # OLS
            error_ols_bootstrap[i] = np.mean(np.mean((z_test.reshape(-1, 1) - zPredict_ols_bootstrap)**2, axis=1, keepdims=True))
            bias_ols_bootstrap[i] = np.mean((z_test.reshape(-1, 1) - np.mean(zPredict_ols_bootstrap, axis=1, keepdims=True))**2)
            var_ols_bootstrap[i] = np.mean(np.var(zPredict_ols_bootstrap, axis=1, keepdims=True))

            # Ridge
            error_ridge_bootstrap[i] = np.mean(np.mean((z_test.reshape(-1, 1) - zPredict_ridge_bootstrap)**2, axis=1, keepdims=True))
            bias_ridge_bootstrap[i] = np.mean((z_test.reshape(-1, 1) - np.mean(zPredict_ridge_bootstrap, axis=1, keepdims=True))**2)
            var_ridge_bootstrap[i] = np.mean(np.var(zPredict_ridge_bootstrap, axis=1, keepdims=True))

            # Lasso
            error_lasso_bootstrap[i] = np.mean(np.mean((z_test.reshape(-1, 1) - zPredict_lasso_bootstrap)**2, axis=1, keepdims=True))
            bias_lasso_bootstrap[i] = np.mean((z_test.reshape(-1, 1) - np.mean(zPredict_lasso_bootstrap, axis=1, keepdims=True))**2)
            var_lasso_bootstrap[i] = np.mean(np.var(zPredict_lasso_bootstrap, axis=1, keepdims=True))

            print("Bootstrap OLS")
            print("degree", "       error", "                Bias", "                  Var")
            print("{}   {}   {}     {}".format(degrees[i], error_ols_bootstrap[i], bias_ols_bootstrap[i], var_ols_bootstrap[i]))


            print("Bootstrap ridge")
            print("{}   {}   {}     {}".format(degrees[i], error_ridge_bootstrap[i], bias_ridge_bootstrap[i], var_ridge_bootstrap[i]))

            print("Bootstrap Lasso")
            print("{}   {}   {}     {}".format(degrees[i], error_lasso_bootstrap[i], bias_lasso_bootstrap[i], var_lasso_bootstrap[i]))

# Plot CVD results
if (cvd == True):
    plt.figure()
    plt.plot(degrees, error_ols_cvd, label="OLS")
    plt.plot(degrees, error_ridge_cvd, label="Ridge")
    plt.plot(degrees, error_lasso_cvd, label= "Lasso")
    plt.title("Cross-validation")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if (bootstrap == True):
    plt.figure()
    plt.plot(degrees, error_lasso_bootstrap, label="OLS")
    plt.plot(degrees, error_ridge_bootstrap, label="Ridge")
    plt.plot(degrees, error_lasso_bootstrap, label="Lasso")
    plt.legend()
    plt.xlabel("Polynomial degree")
    plt.ylabel("error")
    plt.title("Bootstrap")
    plt.show()

if (poly == True):
    plt.figure()
    plt.plot(degrees, mse_ols, label="OLS")
    plt.plot(degrees, mse_ridge, label="Ridge")
    plt.plot(degrees, mse_lasso, label="Lasso")
    plt.legend()
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("MSE evaluation of OLS, Ridge and Lasso. Lambda = {}".format(lmd))
    plt.show()
