'''
Neural network for classification of the Wisconsin breast-cancer data.
Apply the section label "test network" to run simple tests without cross validation
or grid-search.
'''

import numpy as np
import matplotlib.pyplot as plt
from module1 import Sdg, Franke
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd
class NN:
    def __init__(self,
                 X_train,
                 X_test,
                 targets,
                 targets_test,
                 n_hidden_layers,
                 n_hidden_neurons,
                 activation,
                 initilize,
                 printGrad=None):

        self.X_train = X_train
        self.X_test = X_test
        self.t = targets
        self.t_test = targets_test

        self.n_inputs, self.n_features = self.X_train.shape
        self.n_outputs = 1  # Binary classification case
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons

        if(activation == "Sigmoid"):
            self.activation = self.sigmoid
            self.prime = self.prime_sigmoid
        elif(activation == "RELU"):
            self.activation = self.relu
            self.prime = self.prime_relu
        elif(activation == "leaky-RELU"):
            self.activation = self.lrelu
            self.prime = self.prime_lrelu
        else:
            print("Invalid activation function")
            quit()

        self.weights = self.createWeights(initilize)
        self.biases = self.createBiases()

        if printGrad != None:
            self.printGrad = True
        else:
            self.printGrad = False

    def createWeights(self, init):  # Function for creating weight-arrays for all layers
        weights = []

        if (init == "Random"):
            I_w = np.random.randn(self.n_features, self.n_hidden_neurons); weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons))
            O_w = np.random.randn(self.n_hidden_neurons, self.n_outputs); weights.append(O_w)
        elif(init == "Xavier"):
            I_w = np.random.normal(0, np.sqrt(1.0/self.n_features), (self.n_features, self.n_hidden_neurons)); weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.normal(0, np.sqrt(1.0/self.n_hidden_neurons), (self.n_hidden_neurons, self.n_hidden_neurons)))
            O_w = np.random.normal(0, np.sqrt(1.0/self.n_hidden_neurons), (self.n_hidden_neurons, self.n_outputs)); weights.append(O_w)
        elif(init == "He"):
            I_w = np.random.normal(0, np.sqrt(2.0/self.n_features), (self.n_features, self.n_hidden_neurons)); weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.normal(0, np.sqrt(2.0/self.n_hidden_neurons), (self.n_hidden_neurons, self.n_hidden_neurons)))
            O_w = np.random.normal(0, np.sqrt(1.0/self.n_hidden_neurons), (self.n_hidden_neurons, self.n_outputs)); weights.append(O_w)
        else:
            print("Incorrect initilization")
            quit()

        return weights

    def createBiases(self):  # same for biases
        biases = []
        for i in range(0, self.n_hidden_layers):
            biases.append(np.zeros(self.n_hidden_neurons))
        O_b = np.zeros(self.n_outputs)
        biases.append(O_b)
        return biases

    def accuracy_score(self, test, pred):  # Evaluation method
        return np.sum(test == pred) / len(test)

    def sigmoid(self, x):  # Activation function
        return 1.0/(1.0 + np.exp(-x))

    def prime_sigmoid(self, x):
        return self.sigmoid(x)*(1.0-self.sigmoid(x))

    def relu(self, x):
        return x * (x > 0)

    def prime_relu(self, x):
        return 1.0 * (x > 0)

    def lrelu(self, x):
        alpha = 0.01
        y1 = ((x >= 0) * x)
        y2 = ((x < 0) * x * alpha)
        y = y1 + y2
        return y

    def prime_lrelu(self, x):
        alpha = 0.01
        dy1 =((x >= 0) * 1.0)
        dy2 = ((x < 0) * alpha)
        dy = dy1 + dy2
        return dy

    # Method for creating minibatches for SGD
    def create_miniBatches(self, X, y, M):
        mini_batches = []
        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)
        m = data.shape[0] // M
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

    def feed_forward_train(self):  # Forward progation for training the model
        self.activations = []
        self.zs = []
        z_h = np.matmul(self.xi, self.weights[0]) + self.biases[0]; self.zs.append(z_h)
        a_h = self.activation(z_h); self.activations.append(a_h)
        for layer in range(1, self.n_hidden_layers):
            z_h = np.matmul(a_h, self.weights[layer]) + self.biases[layer]; self.zs.append(z_h)
            a_h = self.activation(z_h); self.activations.append(a_h)
        z_o = np.matmul(a_h, self.weights[-1]) + self.biases[-1]; self.zs.append(z_o)
        a_o = self.sigmoid(z_o); self.activations.append(a_o)
        return

    def feed_forward_predict(self, X):  # Final feed forward of a given test set
        z_h = np.matmul(X, self.weights[0]) + self.biases[0]
        a_h = self.activation(z_h)
        for layer in range(1, self.n_hidden_layers):
            z_h = np.matmul(a_h, self.weights[layer]) + self.biases[layer]
            a_h = self.activation(z_h)
        z_o = np.matmul(a_h, self.weights[-1]) + self.biases[-1]
        return self.sigmoid(z_o)

    def back_propagation(self):  # Back propagation algorithm
        def grad(delta_l, layer):
            if (layer == 0):
                self.gradient_weigths_input = np.matmul(self.xi.T, delta_l)
                self.gradient_biases_input = np.sum(delta_l, axis=0)
                if (self.lmd > 0.0):
                    self.gradient_weigths_input += self.lmd * self.weights[layer]
                self.weights[layer] = self.weights[layer] - self.eta * self.gradient_weigths_input
                self.biases[layer] = self.biases[layer] - self.eta * self.gradient_biases_input
            else:
                self.gradient_weigths_hidden = np.matmul(self.activations[layer-1].T, delta_l)
                self.gradient_biases_hidden = np.sum(delta_l, axis=0)
                if (self.lmd > 0.0):
                    self.gradient_weigths_hidden += self.lmd * self.weights[layer]
                self.weights[layer] = self.weights[layer] - self.eta * self.gradient_weigths_hidden
                self.biases[layer] = self.biases[layer] - self.eta * self.gradient_biases_hidden
            return

        delta_l = self.activations[-1] - self.yi.reshape(-1, 1)
        grad(delta_l, -1)
        for layer in range(self.n_hidden_layers-1, 0, -1):
            delta_L = delta_l
            delta_l = np.matmul(delta_L, self.weights[layer+1].T) * self.prime(self.zs[layer])
            grad(delta_l, layer)
        delta_l0 = np.matmul(delta_l, self.weights[1].T) * self.prime(self.zs[0])
        grad(delta_l0, 0)
        return

    # method for training the model, with SGD(Without learning schedule, or other)
    def train(self, n_epochs, M, eta, _lambda):
        acs = []

        self.eta = eta
        self.lmd = _lambda

        for epoch in range(n_epochs):
            mini_batches = self.create_miniBatches(self.X_train, self.t, M)
            for mini_batch in mini_batches:
                self.xi, self.yi = mini_batch
                self.feed_forward_train()
                self.back_propagation()

            if (self.printGrad == True):
                print("Epoch: ", epoch)
                print(self.gradient_weigths_hidden)
                print(self.gradient_weigths_input)

            pred = self.predict(self.X_test)
            acs.append(self.accuracy_score(self.t_test, pred))

        return acs

    def predict(self, X):  # Function for predicting a binary classification set
        y = self.feed_forward_predict(X)
        for i in range(len(y)):
            if y[i] < 0.5:
                y[i] = int(0)
            elif y[i] > 0.5:
                y[i] = int(1)
        return y.ravel()

def accuracy_score(test, pred):  # Evaluation method
    return np.sum(test == pred) / len(test)

# Initilize data

np.random.seed(2021) #Random seed

#Load cancer data from scikit-learn
cancer_data = load_breast_cancer()
X = cancer_data.data
targets = cancer_data.target
scaler = StandardScaler()
X = scaler.fit(X).transform(X)

# Defining the neural network
n_hidden_neurons = 50
n_hidden_layers = 1
activation = "RELU"
initialization = "He"

'''-----------------------------------------------------------------------------
                                Grid-Search
-----------------------------------------------------------------------------'''

n_epochs = 200
M = 10
etas = [1e-4, 1e-3, 1e-2, 5e-2]
lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

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

            network1 = NN(X_train, X_test, t_train, t_test, n_hidden_layers, n_hidden_neurons, activation, initialization)
            network1.train(n_epochs, M, eta, lmd)

            #Predict X_test
            pred = network1.predict(X_test)
            acs_own = accuracy_score(t_test, pred)
            score_own_cvd[cv_split] = acs_own

            #Predict X_train
            #pred_train = network1.predict(X_train)
            #acs_own_train = accuracy_score(t_train, pred_train)
            #score_own_train_cvd[cv_split] = acs_own_train

            #Scikit-learn
            #clf = MLPClassifier(activation="logistic", solver="sgd", max_iter=n_epochs, hidden_layer_sizes=(n_hidden_neurons), batch_size=M, alpha=lmd, learning_rate_init=eta)
            #clf.fit(X_train, t_train)

            #Scikit-learn test
            #t_predict = clf.predict(X_test)
            #acs_scikit = accuracy_score(t_test, t_predict)
            #score_scikit_cvd[cv_split] = acs_scikit

            #SCikit-learn train
            #t_predict_train = clf.predict(X_train)
            #acs_scikit_train = accuracy_score(t_train, t_predict_train)
            #score_scikit_train_cvd[cv_split] = acs_scikit_train

            cv_split += 1

        #Own test
        accuracy_own = np.mean(score_own_cvd); print("eta, lambda: ({}, {}): Own DNN test: {}".format(eta,lmd,accuracy_own))
        acc_grid_own[i][j] = accuracy_own

        #Own train
        #accuracy_own_train = np.mean(score_own_train_cvd); print("itr ({}, {}): Own DNN train: {}".format(i,j,accuracy_own_train))
        #acc_grid_own_train[i][j] = accuracy_own_train

        #scikit test
        #accuracy_scikit = np.mean(score_scikit_cvd); print("itr ({}, {}): Scikit DNN: {}".format(i,j,accuracy_scikit))
        #acc_grid_scikit[i][j] = accuracy_scikit

        #scikit train
        accuracy_scikit_train = np.mean(score_scikit_train_cvd); print("itr ({}, {}): Scikit DNN train: {}".format(i,j,accuracy_scikit_train))
        acc_grid_scikit_train[i][j] = accuracy_scikit_train


        j+= 1
    i += 1
test = pd.DataFrame(acc_grid_own, index = etas, columns = lambdas) #change acc_grid_own to plot other data
fig, ax = plt.subplots()
sns.heatmap(test, annot=True, ax=ax, cmap='viridis', fmt='.4f', vmax=0.98, vmin=0.90)
ax.set_title('Test data accuracy score')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$\eta$')
plt.show()



'''--------------------------------------------------------------------------------
                                #TEST netoworks
--------------------------------------------------------------------------------'''

#Neural network
'''
network1 = NN(X_train, X_test, t_train, t_test, n_hidden_layers, n_hidden_neurons, activation, initialization)  # Create network, printGrad=True prints the gradients for every epoch
network1.train(200, 10, 0.1, 0.000001) #Train network, epochsm batch_size, eta, lambda
pred = network1.predict(X_test) #Predict on new data
acs = accuracy_score(t_test, pred))  # Evalute model
'''

#Scikit-learn neural network
'''
clf = MLPClassifier(hidden_layer_sizes=(100), activation="logistic", solver="sgd", alpha=0.000001, learning_rate_init=0.1, max_iter=200, )
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
print(accuracy_score(t_test, t_predict))
'''
