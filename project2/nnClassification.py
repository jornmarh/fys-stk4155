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
            I_w = np.random.randn(self.n_features, self.n_hidden_neurons)
            weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons))
            O_w = np.random.randn(self.n_hidden_neurons, self.n_outputs)
            weights.append(O_w)
        elif(init == "Xavier"):
            I_w = np.random.randn(self.n_features, self.n_hidden_neurons)*np.sqrt(1.0/(self.n_features))
            weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons) * np.sqrt(1.0/(self.n_hidden_neurons)))
            O_w = np.random.randn(self.n_hidden_neurons, self.n_outputs) * np.sqrt(1.0/(self.n_hidden_neurons))
            weights.append(O_w)
        elif(init == "He"):
            I_w = np.random.randn(self.n_features, self.n_hidden_neurons)*np.sqrt(2.0/(self.n_features))
            weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons) * np.sqrt(2.0/(self.n_hidden_neurons)))
            O_w = np.random.randn(self.n_hidden_neurons, self.n_outputs) * np.sqrt(2.0/(self.n_hidden_neurons))
            weights.append(O_w)
        else:
            print("Incorrect initilization")
            quit()

        return weights

    def createBiases(self):  # same for biases
        biases = []
        for i in range(0, self.n_hidden_layers):
            biases.append(np.zeros(self.n_hidden_neurons) + 0.1)
        O_b = np.zeros(self.n_outputs) + 0.1
        biases.append(O_b)
        return biases

    def sigmoid(self, x):  # Activation function
        print("sigmoid")
        return 1/(1 + np.exp(-x))

    def prime_sigmoid(self, a):
        return a*(1-a)

    def relu(self, x):
        rows, cols = x.shape
        for j in range(rows):
            for k in range(cols):
                if (x[j][k] < 0):
                    x[j][k] = 0
        return x

    def prime_relu(self, x):
        rows, cols = x.shape
        for j in range(rows):
            for k in range(cols):
                if (x[j][k] > 0):
                    x[j][k] = 1
                elif (x[j][k] <= 0):
                    x[j][k] = 0
        return x

    def lrelu(self, x):
        alpha = 0.01
        rows, cols = x.shape
        for j in range(rows):
            for k in range(cols):
                if (x[j][k] <= 0):
                    x[j][k] = alpha*x[j][k]
        return x

    def prime_lrelu(self, x):
        alpha = 0.01
        rows, cols = x.shape
        for j in range(rows):
            for k in range(cols):
                if (x[j][k] > 0):
                    x[j][k] = 1
                elif (x[j][k] <= 0):
                    x[j][k] = alpha
        return x

    def accuracy_score(self, Y_test, Y_pred):  # Evaluation method
        return np.sum(Y_test == Y_pred) / len(Y_test)

    # Method for creating minibatches for SGD
    def create_miniBatches(self, X, y, M):
        mini_batches = []
        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)
        m = data.shape[0] // M
        i = 0
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
                #print(gradient_weigths)
                self.weights[layer] = self.weights[layer] - self.eta * gradient_weigths
                self.biases[layer] = self.biases[layer] - self.eta * gradient_biases
            else:
                gradient_weigths = np.matmul(self.activations[layer-1].T, delta_l)
                gradient_biases = np.sum(delta_l, axis=0)
                if (self.lmd > 0.0):
                    gradient_weigths += self.lmd * self.weights[layer]
                #print(gradient_weigths)
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

    def predict(self, X, t):  # Function for predicting a binary classification set
        y = self.feed_forward_predict(X)
        for i in range(len(y)):
            if y[i] < 0.5:
                y[i] = 0
            elif y[i] > 0.5:
                y[i] = 1
        return self.accuracy_score(t.reshape(-1, 1), y)


# Initilize data
np.random.seed(0)


inputs = load_breast_cancer()
X = inputs.data
y = inputs.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#scaler = StandardScaler()  # Utilizing scikit's standardscaler
#scaler_x = scaler.fit(X_train)  # Scaling x-data
#X_train = scaler_x.transform(X_train)
#X_test = scaler_x.transform(X_test)

print(np.amin(X_train))

# Defining the neural network
n_hidden_neurons = 10
n_hidden_layers = 1

network1 = NN(X_train, y_train, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Random")  # Create network
network1.train(1, X_train.shape[0], 0.01, 0.0001) #Train
#score = network1.predict(X_test, y_test)
#print(score)  # Evalute model

"""
clf = MLPClassifier(activation="logistic", solver="sgd", alpha=0.0001, batch_size=2, learning_rate_init=0.01, max_iter=100)
clf.fit(X_train, y_train)
ytilde = clf.predict(X_test)
print(clf.score(X_test, y_test))
"""
