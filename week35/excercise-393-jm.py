import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error 
import sklearn.linear_model as skl

np.random.seed()
n = 100
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n); x = x.reshape(-1, 1); #print(x); print(x.shape)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

#Desingmatrix of 5th order polynomial
X = np.zeros((len(x), 6))
X[:,0] = 1 
X[:,1] = x[:,0]
X[:,2] = x[:,0]**2
X[:,3] = x[:,0]**3
X[:,4] = x[:,0]**4
X[:,5] = x[:,0]**5

#Pandas display of design matrix
DesignMatrix = pd.DataFrame(X)
DesignMatrix.index = x[:,0]
DesignMatrix.columns = ['1', 'x', 'x^2', 'x^3', 'x^4', 'x^5']
display(DesignMatrix)

#Split data into training and test 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Scale data
scaler = StandardScaler()
scaler_x = scaler.fit(X_train)
scaler_y = scaler.fit(y_train)
X_train_scaled = scaler_x.transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#Unscaled
clf = skl.LinearRegression().fit(X_train, y_train)
ytilde_train = clf.predict(X_train)
ytilde_test = clf.predict(X_test)

#Scaled data
clf_scaled = skl.LinearRegression().fit(X_train_scaled, y_train_scaled)
ytilde_scaled_train = clf_scaled.predict(X_train_scaled)
ytilde_scaled_test = clf_scaled.predict(X_test_scaled)

r2_train = r2_score(y_train, ytilde_train)
r2_test = r2_score(y_test, ytilde_test)
r2_scaled_train = r2_score(y_train_scaled, ytilde_scaled_train)
r2_scaled_test = r2_score(y_test_scaled, ytilde_scaled_test)


mse_train = mean_squared_error(y_train, ytilde_train)
mse_test = mean_squared_error(y_test, ytilde_test)
mse_scaled_train = mean_squared_error(y_train_scaled, ytilde_scaled_train)
mse_scaled_test = mean_squared_error(y_test_scaled, ytilde_scaled_test)


#R2 score:
print("Unscaled data R2 score:", "\n",
      "train = {}, test = {}".format(r2_train, r2_test))
print("Scaled data R2 score:", "\n",
      "train= {}, test = {}".format(r2_scaled_train, r2_scaled_test))

#MSE:

print("Unscaled data MSE score:", "\n",
      "train = {}, test = {}".format(mse_train, mse_test))
print("Scaled data MSE score:", "\n",
      "train= {}, test = {}".format(mse_scaled_train, mse_scaled_test))

