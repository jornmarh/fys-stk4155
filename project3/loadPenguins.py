import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb

def train_test(predictors, targets, out=None):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, test_index in split.split(predictors, targets):
        predictors_train = predictors.loc[train_index]
        predictors_test = predictors.loc[test_index]
        targets_train = targets.loc[train_index]
        targets_test =  targets.loc[test_index]
    if (out==True):
        print("Train: ", predictors_train.shape)
        print("Test: ", predictors_test.shape)
    return predictors_train, predictors_test, targets_train, targets_test

def decisonTree(predictors, targets):
    X_train, X_test, y_train, y_test = train_test(predictors, targets)
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    prediction_tree = tree_clf.predict(X_test)
    print("Single tree: {}".format(accuracy_score(y_test, prediction_tree)))
    return prediction_tree

def bagging(predictors, targets):
    X_train, X_test, y_train, y_test = train_test(predictors, targets)
    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)
    bag_clf.fit(X_train, y_train)
    prediction_bagging = bag_clf.predict(X_test)
    print("Bagging: {}".format(accuracy_score(y_test, prediction_bagging)))
    return prediction_bagging

def randomForest(predictors, targets):
    X_train, X_test, y_train, y_test = train_test(predictors, targets)
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    prediction_rf = rf_clf.predict(X_test)
    print("Random forrest: {}".format(accuracy_score(y_test, prediction_rf)))
    return prediction_rf

def adaboost(predictors, targets):
    X_train, X_test, y_train, y_test = train_test(predictors, targets)
    ab_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    ab_clf.fit(X_train, y_train)
    prediction_ab = ab_clf.predict(X_test)
    print("Adaboost: {}".format(accuracy_score(y_test, prediction_ab)))
    return prediction_ab

def xgboost(predictors, targets):
    X_train, X_test, y_train, y_test = train_test(predictors, targets)
    xg_clf = xgb.XGBClassifier()
    xg_clf.fit(X_train, y_train)
    prediction_xg = xg_clf.predict(X_test)
    print("XGBoost: {}".format(accuracy_score(y_test, prediction_xg)))
    return prediction_xg

np.random.seed(2021)
data = load_penguins()
#Overview
print("Features: \n", data.columns, "\n")
print(data.head(), "\n")
print("Class count: \n", data.groupby('species').count(), "\n")

#Check for nan elements
print("Check for absent values")
print(data.isna().sum())

#Remove rows containing nan elements
data.dropna(inplace=True)
data = data.reset_index()
print("New dataset")
print(data.isna().sum())

body = data.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
penguins = data["species"]
print("\n Fetures for prediction: \n", body); print("\n", "Targets \n", penguins, "\n")


#Classification with a single decision tree
decisonTree(body, penguins)

#Classification with bagging
bagging(body, penguins)

#Classification with random forest
randomForest(body, penguins)

#Classification with AdaBoost
adaboost(body, penguins)

#Classification with XDGBoost
xgboost(body, penguins)
#Classification with neural network
