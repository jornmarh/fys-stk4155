import numpy as np

etas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

scores = np.zeros((len(etas), len(lambdas)))
print(scores.shape)

for eta in etas:
    for lmd in lambdas:
        print('hei')
