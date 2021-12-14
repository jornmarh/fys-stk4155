import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

np.random.seed(2021)

data = load_penguins()
print("Features: \n", data.columns, "\n")

print("Header: \n", data.head(), "\n")

print("Class count: \n", data.groupby('species').count(), "\n")

#Check for nan elements
print("Check for absent values")
print(data.isna().sum())
print(data.shape, "\n")
#Remove rows containing nan elements
data.dropna(inplace=True)
data = data.reset_index()
print("New dataset")
print(data.isna().sum())
print(data.head())

#Split data into body(features) and penguin(species)
body = data.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
print("body")
print(body.head(), "\n")

penguins = data["species"]
counts = penguins.value_counts()
print("penguins")
print(penguins.head(), "\n")
print("Number of counts \n", counts, "\n")

#One-hot encode targets
penguin_dummies = pd.get_dummies(penguins)
print("One hot encoded penguins \n", penguin_dummies.head(), "\n")

#Split into train and test data qith consisistent distribution between classes
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(body, penguin_dummies):
    body_train = body.loc[train_index]
    body_test = body.loc[test_index]
    penguin_train = penguin_dummies.loc[train_index]
    penguin_test =  penguin_dummies.loc[test_index]

'''
print(penguin_test.shape)
print(penguin_train['Adelie'].value_counts())
print(penguin_test['Adelie'].value_counts())
print(penguin_train['Chinstrap'].value_counts())
print(penguin_test['Chinstrap'].value_counts())
print(penguin_train['Gentoo'].value_counts())
print(penguin_test['Gentoo'].value_counts())
'''

#Classification with descission trees.
tree_clf = DecisionTreeClassifier()
tree_clf.fit(body_train, penguin_train)
prediction_tree = tree_clf.predict(body_test)
print("Prediction accuracy: {}".format(accuracy_score(penguin_test, prediction_tree)))

#Classificatio with boosting

#Classification with bagging

#Classification with neural network
