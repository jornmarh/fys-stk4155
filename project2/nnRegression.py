import numpy as np
import matplotlib.pyplot as plt
from module1 import Sdg, Franke
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold

class NN:
    def __init__(self,
                 X_train,
                 X_test,
                 z_train,
                 z_test,
                 n_hidden_layers,
                 n_hidden_neurons,
                 activation,
                 initilize):

        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

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
        elif(activation == "ELU"):
            self.activation = self.elu
            self.prime = self.prime_elu
        else:
            print("Invalid activation function")
            quit()

        self.weights = self.createWeights(initilize)
        self.biases = self.createBiases()

        self.mse = []
        self.r2 = []
        self.epochs = []

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

    def elu(self, x):
        alpha = 1.0
        coef = alpha*(np.exp(x)-1)
        dy1 =((x < 0) * coef)
        dy2 = ((x >= 0) * x)
        dy = dy1 + dy2
        return dy

    def prime_elu(self, x):
        alpha = 1.0
        coef = alpha*(np.exp(x)-1.0)
        dy1 =((x < 0) * np.exp(x))
        dy2 = ((x >= 0) * 1.0)
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
        a_o = z_o; self.activations.append(a_o)

        return

    def feed_forward_predict(self, X):  # Final feed forward of a given test set
        z_h = np.matmul(X, self.weights[0]) + self.biases[0]
        a_h = self.activation(z_h)
        for layer in range(1, self.n_hidden_layers):
            z_h = np.matmul(a_h, self.weights[layer]) + self.biases[layer]
            a_h = self.activation(z_h)
        z_o = np.matmul(a_h, self.weights[-1]) + self.biases[-1]
        return z_o

    def back_propagation(self):  # Back propagation algorithm
        def update(delta_l, layer):
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
        update(delta_l, -1)
        for layer in range(self.n_hidden_layers-1, 0, -1):
            delta_l = np.matmul(delta_l, self.weights[layer+1].T) * self.prime(self.zs[layer])
            update(delta_l, layer)
        delta_l0 = np.matmul(delta_l, self.weights[1].T) * self.prime(self.zs[0])
        update(delta_l0, 0)
        return

    # method for training the model, with SGD(Without learning schedule, or other)
    def train(self, n_epochs, M, eta, _lambda):
        self.eta = eta
        self.lmd = _lambda

        i = 0

        for epoch in range(n_epochs):
            mini_batches = self.create_miniBatches(self.X_train, self.z_train, M)
            for mini_batch in mini_batches:
                self.xi, self.yi = mini_batch
                self.feed_forward_train()
                self.back_propagation()
            ytilde = self.predict(self.X_test)
            mse = mean_squared_error(self.z_test, ytilde)
            r2 = r2_score(self.z_test, ytilde)
            self.epochs.append(i)
            self.mse.append(mse)
            self.r2.append(r2)
            i += 1
            print(i)

    def predict(self, X): #Function for predicting a binary classification set
        y = self.feed_forward_predict(X);
        return y

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)        # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)
    return X

#Initilize data
np.random.seed(64)
N = 20
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
xmesh, ymesh = np.meshgrid(x,y)
xflat = np.ravel(xmesh); yflat = np.ravel(ymesh)

z = (FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N))
X = np.hstack((xflat.reshape(-1,1), yflat.reshape(-1,1)))
#X = create_X(xflat, yflat, 6)
X_train, X_test, z_train, z_test = train_test_split(X,z, test_size=0.2)
scaler = StandardScaler()  # Utilizing scikit's standardscaler
scaler_x = scaler.fit(X_train)  # Scaling x-data
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)

"""
# Initialization if CV is used
z = (FrankeFunction(xflat, yflat) + 0.15*np.random.randn(N*N))
X = np.hstack((xflat.reshape(-1,1), yflat.reshape(-1,1)))
scaler = StandardScaler()
X = scaler.fit(X).transform(X)
"""
# Defining the neural network
n_hidden_neurons = 40
n_hidden_layers = 2
activation = "Sigmoid"
initilize = "Xavier"


"""
# Own dnn vs Scikit
print("Own dnn")
network1 = NN(X_train, z_train, n_hidden_layers, n_hidden_neurons, activation, initilize) #Create network
network1.train(100, 10, 0.001, 0.0000001) #Train
yPredict = network1.predict(X_test)
print(mean_squared_error(z_test.reshape(-1,1), yPredict))
print(r2_score(z_test.reshape(-1,1), yPredict))


print("Scikit dnn")
dnn = MLPRegressor(activation='logistic', solver='sgd', alpha=1e-8, batch_size=10, learning_rate_init=0.001, max_iter=100, random_state=64)
dnn.fit(X_train, z_train)
zPredict = dnn.predict(X_test)
print(mean_squared_error(z_test, zPredict))
print(r2_score(z_test, zPredict))

# MSE as function of epochs with cross-validation and sigmoid activation
cv_split = 0
k = 5
mse = np.zeros((k, 100))
kfold = KFold(n_splits = k, shuffle=True)
for train_indexes, test_indexes in kfold.split(X):
        X_train = X[train_indexes]
        X_test = X[test_indexes]
        z_train = z[train_indexes]
        z_test = z[test_indexes]

        network = NN(X_train, z_train, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Xavier")
        network.train(100, 10, 0.001, 1e-8)
        z_pred = network.predict(X_train)
        mse[cv_split,:] = network.mse

        cv_split += 1

mse = np.mean(mse,axis=0)
plt.plot(network.epochs, mse)
plt.show()


# MSE as function of epochs with sigmoid activation
nn_sig_rand = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Xavier")
nn_sig_rand.train(100, 10, 0.01, 1e-6)
print("MSE: ",nn_sig_rand.mse[-1])
plt.plot(nn_sig_rand.epochs,nn_sig_rand.mse, label='mse')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('mse error')
plt.title('Test mse error as function of epochs with Sigmoid activation')
plt.show()

# r^2 as function of epochs with sigmoid activation
nn_sig_rand = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Xavier")
nn_sig_rand.train(100, 10, 0.01, 1e-6)
print("R2: ",nn_sig_rand.r2[-1])
plt.plot(nn_sig_rand.epochs,nn_sig_rand.r2, label='$r^2$', color='red')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Test $r^2$ error as function of epochs with Sigmoid activation')
plt.show()


# Same, but old version
epochs = np.arange(100)
mse_epochs = np.zeros(len(epochs))
for i in range(len(epochs)):
    nn = NN(X_train, z_train, n_hidden_layers, n_hidden_neurons, activation, initilize)
    nn.train(epochs[i], 10, 0.0001, 1e-7)
    z_pred_epochs = nn.predict(X_test)
    mse_epochs[i] = mean_squared_error(z_test,z_pred_epochs)
    print(mse_epochs[i])

plt.plot(epochs,mse_epochs)
plt.show()




# Gridsearch for eta/lambda
etas = [0.0001,0.0005,0.001,0.005, 0.01]
lambdas = [1e-2, 1e-4, 1e-6, 1e-8]
#mse_grid_train = np.zeros((len(etas),len(lambdas)))
mse_grid_test = np.zeros((len(etas),len(lambdas)))
for i in range(len(etas)):
    for j in range(len(lambdas)):
        nn = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, activation, initilize)
        nn.train(100, 10, etas[i], lambdas[j])
        #z_pred = nn.predict(X_train)
        z_pred_test = nn.predict(X_test)
        #mse_grid_train[i,j] = mean_squared_error(z_train,z_pred)
        mse_grid_test[i,j] = mean_squared_error(z_test,z_pred_test)
        print(mse_grid_test[i,j])

#mse_df_train = pd.DataFrame(mse_grid_train, index = etas, columns = lambdas)
mse_df_test = pd.DataFrame(mse_grid_test, index = etas, columns = lambdas)
fig, ax = plt.subplots(figsize = (7, 7))
sns.heatmap(mse_df_test, annot=True, ax=ax, cmap="viridis_r", fmt='.4f')
ax.set_title("Test error gridsearch")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
plt.show()




# Comparison of Xavier and Random initialisation with Sigmoid activation
nn_sig_xav = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Xavier")
nn_sig_rand = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Random")
nn_sig_xav.train(100, 10, 0.001, 1e-6)
nn_sig_rand.train(100, 10, 0.001, 1e-6)
#print(nn.mse)
#print(nn.epochs)
print("Lowest mse: ",nn_sig_xav.mse[-1])
plt.plot(nn_sig_xav.epochs,nn_sig_xav.mse, label='Xavier')
plt.plot(nn_sig_rand.epochs,nn_sig_rand.mse, label='Random')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('mse')
plt.title('MSE as function of epochs with Sigmoid activation')
plt.show()


"""
# comparison of Sigmoid, Relu, leaku Relu and Elu activation
#np.random.seed(0)
nn_sig_xav = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "Sigmoid", "Xavier")
nn_relu_he = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "RELU", "He")
nn_lrelu_he = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "leaky-RELU", "He")
nn_elu_he = NN(X_train, X_test, z_train, z_test, n_hidden_layers, n_hidden_neurons, "ELU", "He")

nn_sig_xav.train(200, 10, 0.0005, 1e-6)
nn_relu_he.train(200, 10, 0.0005, 1e-6)
nn_lrelu_he.train(200, 10, 0.0005, 1e-6)
nn_elu_he.train(200, 10, 0.0005, 1e-6)
#print(nn.mse)
#print(nn.epochs)
print("Lowest mse sigmoid: ",nn_sig_xav.mse[-1])
print("Lowest mse Relu: ",nn_relu_he.mse[-1])
print("Lowest mse lRelu: ",nn_lrelu_he.mse[-1])
print("Lowest mse ELU: ",nn_elu_he.mse[-1])
plt.plot(nn_sig_xav.epochs,nn_sig_xav.mse, label='Sigmoid & Xavier')
plt.plot(nn_relu_he.epochs,nn_relu_he.mse, label='Relu & He')
plt.plot(nn_lrelu_he.epochs,nn_lrelu_he.mse, label='Leaky-relu & He')
plt.plot(nn_elu_he.epochs,nn_elu_he.mse, label='Elu & He')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('mse')
plt.title('MSE as function of epochs: Comparison of different activations')
plt.show()

print("Lowest $r^2$ Sigmoid: ",nn_sig_xav.r2[-1])
print("Lowest $r^2$ Relu: ",nn_relu_he.r2[-1])
print("Lowest $r^2$ lRelu: ",nn_lrelu_he.r2[-1])
print("Lowest $r^2$ ELU: ",nn_elu_he.r2[-1])
plt.plot(nn_sig_xav.epochs,nn_sig_xav.r2, label='Sigmoid & Xavier')
plt.plot(nn_relu_he.epochs,nn_relu_he.r2, label='Relu & He')
plt.plot(nn_lrelu_he.epochs,nn_lrelu_he.r2, label='Leaky-relu & He')
plt.plot(nn_elu_he.epochs,nn_elu_he.r2, label='Elu & He')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('$r^2 score$')
plt.title('$r^2$ score as function of epochs: Comparison of different activations')
plt.show()
"""


neurons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
layers = [1, 2, 3, 4]
for i in range(len(neurons)):
    for j in range(len(layers)):
        nn_relu_he = NN(X_train, z_train, layers[j], neurons[i], "RELU", "He")
        print("Neurons: ", neurons[i])
        print("Layers: ", layers[j])
        nn_relu_he.train(100, 10, 0.0005, 1e-8)
        print(nn_relu_he.mse[-1])
        print("")
"""
