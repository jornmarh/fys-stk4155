import pandas as pd
import seaborn as sns
from palmerpenguins import load_penguins

sns.set_style("whitegrid")

x, y = load_penguins(return_X_y=True)
print(y)
#print(x)
#print(y.shape)
