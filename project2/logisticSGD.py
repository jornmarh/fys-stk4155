from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import  train_test_split
from autograd import grad
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    rows, cols = x.shape
    for j in range(rows):
        for k in range(cols):
            if (x[j][k] < 0):
                x[j][k] = 0
    return x

def accuracy_score(Y_test, Y_pred): #Evaluation method
    return np.sum(Y_test == Y_pred) / len(Y_test)

def create_miniBatches(X, y, M):
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



cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)



X = np.array([ [0,0], [0,1], [1,0],[1,1], [0,0], [0,1], [1,0], [1,1] ])
yXOR = np.array( [ 0, 1 ,1, 0, 0, 1, 1, 0] )
X_train, X_test, y_train, y_test = train_test_split(X,yXOR)

np.random.seed(64)
theta = np.random.randn(X.shape[1])
epochs = 100
eta = 0.001
M = 1
n = int(X.shape[0] / M)

costfunction_best = 100

for i in range(epochs):
    mini_batches = create_miniBatches(X_train, y_train, M)
    for mini_batch in mini_batches:
        xi,yi = mini_batch
        g = 2*xi.T @ ((xi @ theta) - yi)
        #g = - xi.T @ (yi - p) #dC/dB, obtained from the maximum likelihood estimation
        #p = np.exp(t)/(1+np.exp(t))
        #g = -xi.T @ (yi-p)
        theta = theta - eta*g
        t = xi @ theta
        costfunction = - (yi @ t - np.log(1 + np.exp(t))) # Cross-entropy
        costfunction = np.sum(costfunction)
        #prob = np.exp(t)/(1+np.exp(t))

        if (costfunction) < (costfunction_best):
            costfunction_best = costfunction
            theta = theta

        #theta -= eta*gs

ytilde_lin = X_train @ theta
y_tilde_log = sigmoid(ytilde_lin)
#print(y_tilde_log)

for i in range(len(y_tilde_log)):
    if y_tilde_log[i] < 0.5:
        y_tilde_log[i] = 0
    elif y_tilde_log[i] >= 0.5:
        y_tilde_log[i] = 1

print(yXOR)
print (y_tilde_log)

score = accuracy_score(y_train,y_tilde_log)
print(score)

# Scikit
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
ypred = logreg.predict(X_train)
print(logreg.score(X_train, y_train))
print(ypred)
