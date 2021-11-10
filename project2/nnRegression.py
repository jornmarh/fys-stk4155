import numpy as np
import matplotlib.pyplot as plt
from module1 import Sdg, Franke
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


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
        self.n_outputs = 1 #Binary classification case
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons

        if(activation == "sigmoid"):
            self.activation = self.sigmoid
        elif(activation == "elu"):
            self.activation = self.relu
        elif(activation == "lrelu"):
            self.activation = self.lrelu

        self.weights = self.createWeights(initilize)
        self.biases = self.createBiases(initilize)


    def createWeights(self, init): #Function for creating weight-arrays for all layers
        weights = []
        if (init == "normal"):
            I_w = np.random.randn(self.n_features, self.n_hidden_neurons)
            weights.append(I_w)
            for i in range(1, self.n_hidden_layers):
                weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons))
            O_w = np.random.randn(self.n_hidden_neurons, self.n_outputs)
            weights.append(O_w)

        return weights

    def createBiases(self, inti): #same for biases
        biases = []
        if (init == "normal"):
            for i in range(0, self.n_hidden_layers):
                biases.append(np.zeros(self.n_hidden_neurons) + 0.01)
            O_b = np.zeros(self.n_outputs)
            biases.append(O_b)

        return biases

    def sigmoid(self, x): #Activation function
        return 1/(1 + np.exp(-x))
    def elu(self, x):
        alpha = 1
        if (x < 0):
            return alpha*(np.exp(x) - 1)
        else:
            return x
    def lRelu(self, x):
        alpha = 0.01
        if (x < 0):
            return alpha*x
        else:
            return x

    def accuracy_score(self, Y_test, Y_pred): #Evaluation method
        return np.sum(Y_test == Y_pred) / len(Y_test)

    def create_miniBatches(self, X, y, M): #Method for creating minibatches for SGD
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

    def feed_forward_train(self): #Forward progation for training the model
        self.activations = []
        z_h = np.matmul(self.xi, self.weights[0]) + self.biases[0]
        a_h = self.sigmoid(z_h); self.activations.append(a_h)
        for layer in range(1, self.n_hidden_layers):
            z_h = np.matmul(a_h, self.weights[layer]) + self.biases[layer]
            a_h = self.sigmoid(z_h); self.activations.append(a_h)
        z_o = np.matmul(a_h, self.weights[-1]) + self.biases[-1]
        a_o = z_o; self.activations.append(a_o)
        return

    def feed_forward_predict(self, X): #Final feed forward of a given test set
        z_h = np.matmul(X, self.weights[0]) + self.biases[0]
        a_h = self.sigmoid(z_h)
        for layer in range(1, self.n_hidden_layers):
            z_h = np.matmul(a_h, self.weights[layer]) + self.biases[layer]
            a_h = self.sigmoid(z_h)
        z_o = np.matmul(a_h, self.weights[-1]) + self.biases[-1]
        return z_o

    def back_propagation(self): #Back propagation algorithm
        def grad(delta_l, layer):
            if (layer == 0):
                gradient_weigths = np.matmul(self.xi.T, delta_l)
                gradient_biases = np.sum(delta_l, axis=0)
                if (self.lmd > 0.0):
                    gradient_weigths += self.lmd * self.weights[layer]
                self.weights[layer] = self.weights[layer] - self.eta * gradient_weigths
                self.biases[layer] = self.biases[layer] - self.eta * gradient_biases
            else :
                gradient_weigths = np.matmul(self.activations[layer-1].T, delta_l)
                gradient_biases = np.sum(delta_l, axis=0)
                if (self.lmd > 0.0):
                    gradient_weigths += self.lmd * self.weights[layer]
                self.weights[layer] = self.weights[layer] - self.eta * gradient_weigths
                self.biases[layer] = self.biases[layer] - self.eta * gradient_biases
            return

        delta_l = self.activations[-1] - self.yi.reshape(-1,1)
        grad(delta_l, -1)
        for layer in range(self.n_hidden_layers-1, 0, -1):
            delta_L = delta_l
            delta_l = np.matmul(delta_L, self.weights[layer+1].T) * self.activations[layer]*(1-self.activations[layer])
            grad(delta_l, layer)
        delta_l0 = np.matmul(delta_l, self.weights[1].T) * self.activations[0]*(1-self.activations[0])
        grad(delta_l0,0)

        return

    def train(self, n_epochs, M, eta, _lambda): #method for training the model, with SGD(Without learning schedule, or other)
        self.eta = eta
        self.lmd = _lambda
        for epoch in range(n_epochs):
            mini_batches = self.create_miniBatches(self.X_train, self.t, M)
            for mini_batch in mini_batches:
                self.xi, self.yi = mini_batch
                self.feed_forward_train()
                self.back_propagation()

    def predict(self, X, t): #Function for predicting a binary classification set
        y = self.feed_forward_predict(X);
        return mean_squared_error(t, y)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

#Initilize data
np.random.seed(64)
N = 20
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh); yflat = np.ravel(ymesh)

z = (FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N))
X = np.hstack((xflat.reshape(-1,1), yflat.reshape(-1,1)))

X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

# Defining the neural network
n_hidden_neurons = 100
n_hidden_layers = 3
activation = "sigmoid"

network1 = NN(X_train, z_train, n_hidden_layers, n_hidden_neurons, activation) #Create network
network1.train(100, 5, 0.005, 0.0001) #Train
score = network1.predict(X_test, z_test); print(score) #Evalute model

clf = MLPRegressor(activation='logistic', solver='sgd', alpha=0.0001, batch_size=5, learning_rate_init=0.005, max_iter=100, random_state=0)
clf.fit(X_train, z_train)
p = clf.predict(X_test)
print(mean_squared_error(z_test, p))
