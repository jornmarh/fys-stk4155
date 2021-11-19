from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def sigmoid(x):
    return 1/(1+np.exp(-x))

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

def predict(X_test, weights):
    prediction = sigmoid(X_test @ weights)
    for i in range(len(prediction)):
        if (prediction[i] < 0.5):
            prediction[i] = 0
        elif (prediction[i] > 0.5):
            prediction[i] = 1
    return prediction

def sgd_logreg(X_train, y_train):
    n = X_train.shape[1]
    n_epochs = 200
    M = 10
    lmd = 0
    weights = np.random.randn(n)
    for i in range(n_epochs):
        mini_batches = create_miniBatches(X_train, y_train, M)
        for mini_batch in mini_batches:
            xi, yi = mini_batch
            gradient = -xi.T @ (yi - sigmoid(xi@weights)) + lmd*weights
            weights -= gradient
    return weights


#Load cancer data from scikit-learn
np.random.seed(2021)
cancer_data = load_breast_cancer()
X = cancer_data.data
targets = cancer_data.target
scaler = StandardScaler()
X = scaler.fit(X).transform(X)

k = 10
kfold = KFold(n_splits = k, shuffle=True)
scores_scikit = np.zeros(k)
scores_own = np.zeros(k)
cv_split = 0
for train_indexes, test_indexes in kfold.split(X):
    X_train = X[train_indexes]
    X_test = X[test_indexes]
    t_train = targets[train_indexes]
    t_test = targets[test_indexes]

    acc_own = accuracy_score(t_test, predict(X_test, sgd_logreg(X_train, t_train))); print('Own:     split: {}, score: {}'.format(cv_split, acc_own))

    logreg = LogisticRegression(penalty = 'l2', max_iter = 200)
    logreg.fit(X_train, t_train)
    scikit_pred = logreg.predict(X_test)
    acc_scikit = accuracy_score(t_test, scikit_pred); print('Scikit:    split: {}, score: {}'.format(cv_split, acc_scikit))

    scores_scikit[cv_split] = acc_scikit
    scores_own[cv_split] = acc_own

    cv_split += 1

score_scikit = np.mean(scores_scikit)
score_own = np.mean(scores_own)
print('Score from own method: {}, score from scikit-learn: {}'.format(score_own, score_scikit))















'''for epoch in range(n_epochs):
    mini_batches = create_miniBatches(X, targets, 10)
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

        theta -= eta*g'''
