import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def x2(x):
	return x**2

x = 6

activation = "x2"

if (activation == "sigmoid"):
	activation = sigmoid
elif (activation == "x2"):
	activation = x2
print(activation(x))
