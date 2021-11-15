import numpy as np

neurons = np.arange(20,300)
neuron_score_own = np.zeros(len(neurons))
neuron_score_scikit = np.zeros(len(neurons))
i = 0
for n_hidden_neurons in neurons:
    print(n_hidden_neurons)
    i+=1
