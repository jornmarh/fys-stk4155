import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from sklearn import datasets
from module1 import Sdg

class NN:
    def __init__(self,
                X_train,
                targets,
                hidden_weights,
                hidden_bias,
                output_weights,
                output_bias):
        self.X_train = X_train
        self.t = targets
        self.w_h = hidden_weights
        self.b_h = hidden_bias
        self.w_o = output_weights
        self.b_o = output_bias

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def accuracy_score(self, Y_test, Y_pred):
        return np.sum(Y_test == Y_pred) / len(Y_test)

    def create_miniBatches(self, X, y, M):
        mini_batches = []
        data = np.hstack((X, y.reshape(-1,1)))
        np.random.shuffle(data)
        m = data.shape[0] // M
        i = 0

        for i in range(m + 1):
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

    def feed_forward(self): #One hidden layer
        z_h = np.matmul(self.xi, self.w_h) + self.b_h
        self.a_h = self.sigmoid(z_h)

        z_o = np.matmul(self.a_h, self.w_o) + self.b_o
        self.a_o = self.sigmoid(z_o)
        return

    def forwardPredict(self, X):
        z_h = np.matmul(X, self.w_h) + self.b_h
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.w_o) + self.b_o
        a_o = self.sigmoid(z_o)

        return a_o

    def back_propagation(self): #One hidden layer
        delta_L =  self.a_o - self.yi.reshape(-1,1)
        delta_l = np.matmul(delta_L, self.w_o.T) * self.a_h * (1 - self.a_h)

        gradient_w_o = np.matmul(self.a_h.T, delta_L)
        gradient_b_o = np.sum(delta_L, axis=0)

        gradient_w_h = np.matmul(self.xi.T, delta_l)
        gradient_b_h = np.sum(delta_l, axis=0)

        if (self.lmd > 0.0):
            gradient_w_o += self.lmd * self.w_o
            gradient_w_h += self.lmd * self.w_h

        self.w_o = self.w_o - self.eta * gradient_w_o
        self.b_o = self.b_o - self.eta * gradient_b_o
        self.w_h = self.w_h - self.eta * gradient_w_h
        self.b_h = self.b_h - self.eta * gradient_b_h

        return

    def train(self, n_epochs, eta, M, _lambda):
        self.eta = eta
        self.lmd = _lambda
        for epoch in range(n_epochs):
            mini_batches = self.create_miniBatches(self.X_train, self.t, M)
            for mini_batch in mini_batches:
                self.xi, self.yi = mini_batch
                self.feed_forward()
                self.back_propagation()

    def predict(self, X, t):
        y = self.forwardPredict(X); print(y)
        for i in range(len(y)):
            if y[i] < 0.5:
                y[i] = 0
            elif y[i] > 0.5:
                y[i] = 1
        return self.accuracy_score(t.reshape(-1,1), y)


np.random.seed(0)
# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
yXOR = np.array( [ 0, 1 ,1, 0])

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_outputs = 1
n_hidden_layers = 1

hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01
output_weights = np.random.randn(n_hidden_neurons, n_outputs)
output_bias = np.zeros(n_outputs) + 0.01

network1 = NN(X, yXOR, hidden_weights, hidden_bias, output_weights, output_bias)
network1.train(1000, 0.5, 18, 0.0001)
score = network1.predict(X, yXOR)
print(score)
