import numpy as np
np.random.seed(0)
# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
yXOR = np.array( [ 0, 1 ,1, 0])

a = np.random.randn(10)

print(a)
for i in range(len(a)-1, 0, -1):
    print(i)
