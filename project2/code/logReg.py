'''
Code for logistic regression
'''


from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd

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

def sgd_logreg(X_train, y_train, eta, lmd):
    n = X_train.shape[1]
    n_epochs = 200
    M = 10
    weights = np.random.randn(n)
    for i in range(n_epochs):
        mini_batches = create_miniBatches(X_train, y_train, M)
        for mini_batch in mini_batches:
            xi, yi = mini_batch
            gradient = -xi.T @ (yi - sigmoid(xi@weights)) + lmd*weights
            weights -= eta*gradient
    return weights


#Load cancer data from scikit-learn
np.random.seed(2021)
cancer_data = load_breast_cancer()
X = cancer_data.data
targets = cancer_data.target
scaler = StandardScaler()
X = scaler.fit(X).transform(X)


#Heatmap
n_epochs = 200
M = 10
etas = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
lambdas = [1e-8, 1e-6, 1e-4, 1e-1, 1]

k=10
kfold = KFold(n_splits = k, shuffle=True)

score_own_cvd = np.zeros(k)
score_own_train_cvd = np.zeros(k)
score_scikit_cvd = np.zeros(k)
score_scikit_train_cvd = np.zeros(k)

acc_grid_own = np.zeros((len(etas), len(lambdas)))
acc_grid_own_train = np.zeros((len(etas), len(lambdas)))
acc_grid_scikit = np.zeros((len(etas), len(lambdas)))
acc_grid_scikit_train = np.zeros((len(etas), len(lambdas)))

sns.set()

i = 0
for eta in etas:
    j = 0
    for lmd in lambdas:
        cv_split = 0
        for train_indexes, test_indexes in kfold.split(X):
            X_train = X[train_indexes]
            X_test = X[test_indexes]
            t_train = targets[train_indexes]
            t_test = targets[test_indexes]

            #acc_own = accuracy_score(t_test, predict(X_test, sgd_logreg(X_train, t_train, eta, lmd))); print('Own:     split: {}, score: {}'.format(cv_split, acc_own))
            #acc_own_train = accuracy_score(t_train, predict(X_train, sgd_logreg(X_train, t_train, eta, lmd))); print('Own:     split: {}, score: {}'.format(cv_split, acc_own_train))

            #score_own_cvd[cv_split] = acc_own
            #score_own_train_cvd[cv_split] = acc_own_train


            cv_split += 1

        #accuracy_own = np.mean(score_own_cvd)
        #acc_grid_own[i][j] = accuracy_own

        #accuracy_own_train = np.mean(score_own_train_cvd)
        #acc_grid_own_train[i][j] = accuracy_own_train


        j+= 1
    i += 1
test = pd.DataFrame(acc_grid_own_train, index = etas, columns = lambdas)
fig, ax = plt.subplots()
sns.heatmap(test, annot=True, ax=ax, cmap='viridis', fmt='.4f')
ax.set_title('Test data accuracy score')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\eta$')
plt.show()

#Test data
'''
k = 10
kfold = KFold(n_splits = k, shuffle=True)
scores_scikit = np.zeros(k)
scores_own = np.zeros(k)
cv_split = 0

lmd = 1e-3
eta = 1e-1

for train_indexes, test_indexes in kfold.split(X):
    X_train = X[train_indexes]
    X_test = X[test_indexes]
    t_train = targets[train_indexes]
    t_test = targets[test_indexes]

    acc_own = accuracy_score(t_test, predict(X_test, sgd_logreg(X_train, t_train, eta, lmd))); print('Own:     split: {}, score: {}'.format(cv_split, acc_own))

    clf = SGDClassifier(loss='log', penalty='l2', max_iter=200, alpha=lmd, eta0=eta)
    clf.fit(X_train, t_train)
    predict_scikit = clf.predict(X_test)
    acc_scikit = accuracy_score(t_test, predict_scikit)


    scores_scikit[cv_split] = acc_scikit
    scores_own[cv_split] = acc_own

    cv_split += 1

score_scikit = np.mean(scores_scikit)
score_own = np.mean(scores_own)
print('Score from own method: {}, score from scikit-learn: {}'.format(score_own, score_scikit))
'''
