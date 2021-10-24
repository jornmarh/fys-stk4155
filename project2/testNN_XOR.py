import numpy as np

n_hiddenLayers = 1
n_hiddenNodes = 50
n_inputs = X.shape[1]
n_outputs = 1
n_categories =

hidden_weights
hidden_bias

output_weigths
output_bias

def sigmoid(x):
    return 1/(1+np.exp(x))


def feed_forward(a_h = None):
    iter = 0
    while iter < n_hiddenLayers:
        if (a_h == None):
            z_h =
        else:
            z_h = a_h*W + b
        a_h = sigmoid(z_h)
        iter += 1
        feed_forward(a_h)
    feed_forward_output(a_h):

def feed_forward_output(a_h):
    return "test"
