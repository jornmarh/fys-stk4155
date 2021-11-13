from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import  train_test_split
from autograd import grad


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

X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]])
yXOR = np.array( [ 0, 1 ,1, 0])

"""
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
"""

np.random.seed(64)
theta = np.random.randn(X.shape[1])
epochs = 100
eta = 0.01
M = 1
n = int(X.shape[0] / M)

costfunction_best = 100
for i in range(epochs):
    mini_batches = create_miniBatches(X, yXOR, M)
    for mini_batch in mini_batches:
        xi,yi = mini_batch
        t = xi @ theta
        costfunction = yi.dot(t) - np.log( 1 + np.exp(t))
        #print(costfunction)
        costfunction = -np.sum(costfunction)
        p = sigmoid(t)
        g = - xi.T @ (yi - p) #dC/dB, obtained from the maximum likelihood estimation

        if (costfunction) < (costfunction_best):
            costfunction_best = costfunction
            best_theta = theta

        theta -= eta*g


ytilde_lin = X @ theta
y_tilde_log = sigmoid(ytilde_lin)

for i in range(len(y_tilde_log)):
    if y_tilde_log[i] < 0.5:
        y_tilde_log[i] = 0
    elif y_tilde_log[i] >= 0.5:
        y_tilde_log[i] = 1

print (y_tilde_log)

score = accuracy_score(yXOR,y_tilde_log)
print(score)




        #cost = - self.y_train @ t - np.log(1 + np.exp(t))


"""
steg 1: sigmoid med xavier: se på mse som funksjon av layers og neurons . før hyperparameters
steg 2: sammenlign med MLPRegressor

Steg 3: gridsearch for lamda og eta, epoch analyse
Steg 4: Analyse med forskjellige aktiveringsfunksjonene
optional: Kan se hvor fort forskjellige initialiseringer konvergerer

"""
