import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cost(a, t):
    return -(t * np.log( a ) + (1-t)*np.log( 1 - a ))


def feed_forward(X):
    layer = 1
    print("hidden layer ", str(layer))
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    a_h = sigmoid(z_h)
    while (layer < n_hidden_layers):
        layer += 1
        print("hidden layer: ", str(layer))
        z_h = np.matmul(a_h, hidden_weights) + hidden_bias
        a_h = sigmoid(z_h)
    print("output layer")
    z_o = np.matmul(a_h, output_weights) + output_bias
    probaibility = sigmoid(z_o)

    return probaibility

def back_propagation():
    return "uferdig"

def predict(a_0):
    return np.argmax(a_o, axis=1)

np.random.seed(0)

# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
yXOR = np.array( [ 0, 1 ,1, 0])

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 2
n_hidden_layers = 1

hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

a_o = feed_forward(X)
print(a_o)
print(str(predict(a_o)))
