import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


def feed_forward(X):
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    a_h = sigmoid(z_h)

    z_o = np.matmul(a_h, output_weights) + output_bias
    a_o = sigmoid(z_o)

    return a_o

def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)

np.random.seed(0)
# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

n_inputs, n_features = X.shape
n_categories = 2
n_hidenNeurons = 2

hidden_weights = np.random.rand(n_features, n_hidenNeurons)
hidden_bias = np.zeros(n_hidenNeurons) + 0.01

output_weights = np.random.rand(n_hidenNeurons, n_categories)
output_bias =  np.zeros(n_categories) + 0.01

probs = feed_forward(X)
print(probs)

predictions = predict(X)
print(predictions)
