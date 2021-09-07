#Excercise 3.9.2 Jørn-Marcus
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)

#print(x.shape)
#print(x)

X = np.zeros((len(x),3))
X[:,0] = 1
X[:,1] = x[:,0]
X[:,2] = x[:,0]**2


beta = np.linalg.inv(X.T @ X) @ X.T @ y
y_tilde = X @ beta

fig, ax = plt.subplots()
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.scatter(x, y, alpha=0.7,lw=2, label='data')
ax.scatter(x, y_tilde, alpha=0.7, lw=2, c = 'm', label = 'fit')

ax.legend()
plt.show()
