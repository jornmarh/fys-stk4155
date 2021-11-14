import numpy as np
import matplotlib.pyplot as plt
from module1 import Sdg, Franke
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer

class NN:
    def __init__(self,
                 X_train,
                 targets,
                 n_hidden_layers,
                 n_hidden_neurons,
                 activation,
                 initilize):

        self.X_train = X_train
        self.t = targets

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
                gradient_weigths = np.matmul(self.xi.T, delta_l)
                gradient_biases = np.sum(delta_l, axis=0)

                if (self.lmd > 0.0):
                    gradient_weigths += self.lmd * self.weights[layer]
            
                self.weights[layer] = self.weights[layer] - self.eta * gradient_weigths
                self.biases[layer] = self.biases[layer] - self.eta * gradient_biases
            else:
                gradient_weigths = np.matmul(self.activations[layer-1].T, delta_l)
                gradient_biases = np.sum(delta_l, axis=0)

                if (self.lmd > 0.0):
                    gradient_weigths += self.lmd * self.weights[layer]

                self.weights[layer] = self.weights[layer] - self.eta * gradient_weigths
                self.biases[layer] = self.biases[layer] - self.eta * gradient_biases
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
        self.eta = eta
        self.lmd = _lambda
        for epoch in range(n_epochs):
            mini_batches = self.create_miniBatches(self.X_train, self.t, M)
            for mini_batch in mini_batches:
                self.xi, self.yi = mini_batch
                self.feed_forward_train()
                self.back_propagation()

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
np.random.seed(64)

cancer_data = load_breast_cancer()
X = cancer_data.data
targets = cancer_data.target

'''
temp1=np.reshape(inputs[:,1],(len(inputs[:,1]),1))
temp2=np.reshape(inputs[:,2],(len(inputs[:,2]),1))
X=np.hstack((temp1,temp2))
temp=np.reshape(inputs[:,5],(len(inputs[:,5]),1))
X=np.hstack((X,temp))
temp=np.reshape(inputs[:,8],(len(inputs[:,8]),1))
X=np.hstack((X,temp))
print(X.shape)
del temp1,temp2,temp
'''

X_train, X_test, t_train, t_test = train_test_split(X,targets, test_size=0.1)

scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# Defining the neural network
n_hidden_neurons = 200
n_hidden_layers = 2

activation = "RELU"
initialization = "He"

network1 = NN(X_train, t_train, n_hidden_layers, n_hidden_neurons, activation, initialization)  # Create network
network1.train(20, 100, 0.0001, 0.001) #Train
pred = network1.predict(X_test)
print(pred-t_test)
print(accuracy_score(t_test, pred))  # Evalute model

clf = MLPClassifier(activation="relu", solver="sgd", alpha=0.001, batch_size=10, learning_rate_init=0.0001, max_iter=200, random_state=0)
clf.fit(X_train, t_train)
t_predict = clf.predict(X_test)
print(accuracy_score(t_test, t_predict))
