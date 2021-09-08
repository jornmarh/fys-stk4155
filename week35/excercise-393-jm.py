import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.linear_model as skl

def funksjonnavnet(degree) :
    degree+=1
    s = []
    #Desingmatrix of 5th order polynomial
    X = np.zeros((len(x), degree))
    for i in range(degree):
        X[:,i] = x[:,0]**i
        s.append("x^{}".format(i))

    '''
#Pandas display of design matrix
    DesignMatrix = pd.DataFrame(X)
    DesignMatrix.index = x[:,0]
    DesignMatrix.columns = s
    display(DesignMatrix)
    '''

    #Split data into training and test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    clf = skl.LinearRegression().fit(X_train, y_train)
    ytilde_train = clf.predict(X_train)
    ytilde_test = clf.predict(X_test)

    #R2-score
    r2_train = r2_score(y_train, ytilde_train)
    r2_test = r2_score(y_test, ytilde_test)

    #MSE error
    mse_train = mean_squared_error(y_train, ytilde_train)
    mse_test = mean_squared_error(y_test, ytilde_test)

    return r2_train, r2_test, mse_train, mse_test


    #print("Unscaled data R2 score:", "\n",
    #      "train = {}, test = {}".format(r2_train, r2_test))

    #MSE:
    #print("Unscaled data MSE score:", "\n",
    #      "train = {}, test = {}".format(mse_train, mse_test))

np.random.seed()
n = 100
# Make data set.
x = np.linspace(-3, 3, n); x = x.reshape(-1, 1);
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

maxdegree = 15
degrees = np.zeros(maxdegree)
r2_train_list = np.zeros(maxdegree)
r2_test_litst = np.zeros(maxdegree)
mse_train_list = np.zeros(maxdegree)
mse_test_list = np.zeros(maxdegree)

for i in range(maxdegree):
    r2_train, r2_test, mse_train, mse_test = funksjonnavnet(i)
    r2_train_list[i] = r2_train
    r2_test_litst[i] = r2_test
    mse_train_list[i] = mse_train
    mse_test_list[i] = mse_test
    degrees[i] = i

                                #Plot Data
fig, ax = plt.subplots()
ax.set_xlabel('Number of degrees')
ax.set_ylabel('Error')
ax.plot(degrees, mse_train_list, alpha=0.7,lw=2, c = 'red', label='MSE_train')
ax.plot(degrees, mse_test_list, alpha=0.7, lw=2, c = 'blue', label='MSE_test')
ax.plot(degrees, r2_train_list, alpha=0.7, lw=2, c = 'brown', label='R2_train')
ax.plot(degrees, r2_test_litst, alpha=0.7, lw=2, c = 'red', label='R2_test')
ax.legend()
plt.show()


'''
                                #Scale data

scaler = StandardScaler()
scaler_x = scaler.fit(X_train)
#scaler_y = scaler.fit(y_train)
X_train_scaled = scaler_x.transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
scaler_y = scaler.fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#Unscaled

#Scaled data
clf_scaled = skl.LinearRegression().fit(X_train_scaled, y_train_scaled)
ytilde_scaled_train = clf_scaled.predict(X_train_scaled)
ytilde_scaled_test = clf_scaled.predict(X_test_scaled)

r2_scaled_train = r2_score(y_train_scaled, ytilde_scaled_train)
r2_scaled_test = r2_score(y_test_scaled, ytilde_scaled_test)


mse_scaled_train = mean_squared_error(y_train_scaled, ytilde_scaled_train)
mse_scaled_test = mean_squared_error(y_test_scaled, ytilde_scaled_test)

print("Scaled data R2 score:", "\n",
"train= {}, test = {}".format(r2_scaled_train, r2_scaled_test))

print("Scaled data MSE score:", "\n",
"train= {}, test = {}".format(mse_scaled_train, mse_scaled_test))
'''
