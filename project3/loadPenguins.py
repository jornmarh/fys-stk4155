import pandas as pd
import numpy as np
import seaborn as sns
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score, cross_val_predict
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import tree
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import LabelEncoder
import dataframe_image as dfi

def find_best(model, variables, type='depth'):
    '''
    This is a function to find the optimal value of a model, that could be
    the learning rate, max_depth, or other. The argumets of this function is:
    1) the model to optimize in stringformat, for example "Decisiontree"
    2) A list of variables, for max_depth this could be [1,2,3,4,5]
    3) An optional parameter, type. This is for model="Decisiontree",
        type="a" means optimze pruning parameter
    The function returns the best variable in the set, and corresponding score for that variable
    '''
    i=0
    best_score=0
    best_variable=0
    for v in variables:
        if (model=="Randomforest"):
            clf = RandomForestClassifier(n_estimators=200, max_depth=v)
        elif (model=="Decisiontree"):
            if (type=='a'):
                clf = DecisionTreeClassifier(criterion='gini', max_depth=6, ccp_alpha=v)
            else:
                clf = DecisionTreeClassifier(criterion='gini', max_depth=v)
        elif (model=='Adaboost'):
            clf = AdaBoostClassifier(n_estimators=200, learning_rate=v)
        score_v = np.mean(cross_val_score(clf, body, penguins, cv=10))
        if (score_v > best_score):
            best_score = score_v
            best_variable = v
        i+=1
    return best_variable, best_score

def train_test(predictors, targets, size=0.2, out=None):
    '''
    This method is used for generating a train test split of the data,
    using scikit-learns StratifiedShuffleSplit method
    The arguments are
    4) out: set to True to print information about the split
    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=size)
    for train_index, test_index in split.split(predictors, targets):
        predictors_train = predictors.loc[train_index]
        predictors_test = predictors.loc[test_index]
        targets_train = targets.loc[train_index]
        targets_test =  targets.loc[test_index]
    if (out==True):
        print("Train shape: ", predictors_train.shape)
        print("Training class count ratio: \n", targets_train.value_counts()/len(targets_train))
        print("Test shape: ", predictors_test.shape)
        print("Test class count ratio: \n", targets_test.value_counts()/len(targets_test))
    return predictors_train, predictors_test, targets_train, targets_test

def plots(y_test, y_pred, y_proba):
    '''
    Takes test data, predictions and probabilities from a model to plot the
    confusion matrix, precision_recall curbes and roc.
    '''
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    skplt.metrics.plot_precision_recall(y_test, y_proba)
    skplt.metrics.plot_roc(y_test, y_proba)
    plt.show()

def bias_var(model, X_train, X_test, y_train, y_test, n_rounds, best_estimator=None, m_depth=12):
    '''
    This function returns the bias and variance of the model, along with a plot
    as a function of the tree depth.
    The targets are label encoded with scikits LabelEncoder
    arguments are
    6) n_rounds = number of bootstrap for the bias_var tradeoff
    7) best_estimator = (optional) what tree depth to return bias and var from
    8) m_depth = (optional) max value for depth the analysis is done for
    '''
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    bias = np.zeros(m_depth)
    var = np.zeros(m_depth)
    depth = np.zeros(m_depth)
    bias_return = 0
    var_return = 0
    i=0
    for m in range(1, m_depth):
        if (model=="Decisiontree"):
            clf = DecisionTreeClassifier(max_depth=m)
        elif (model=="Randomforest"):
            clf = RandomForestClassifier(n_estimators=100, max_depth=m)
        elif (model=="Bagging"):
            clf = BaggingClassifier(DecisionTreeClassifier(max_depth=m), n_estimators=100)
        elif (model=="Ada"):
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m), n_estimators=50, learning_rate=0.5)
        elif (model=="XGB"):
            clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=m, verbosity=0)

        l, b, v = bias_variance_decomp(clf, X_train.values, y_train, X_test.values, y_test, num_rounds=n_rounds, random_seed=2021)
        bias[i] = b
        var[i] = v
        depth[i] = m
        if (m == best_estimator):
            bias_return=b
            var_return=v
        i+=1
    plt.scatter(depth, bias, label='bias')
    plt.scatter(depth, var, label='variance')
    plt.legend()
    plt.title("Bias variance tradeoff of tree depth")
    plt.xlabel("Tree depth")
    plt.ylabel("Error")
    plt.xlim(0, m_depth)
    plt.show()
    return bias_return, var_return

def decisonTree(model, X_train, X_test, y_train, y_test):
    tree_clf = model
    tree_clf.fit(X_train, y_train)
    prediction_tree = tree_clf.predict(X_test)
    probability_tree = tree_clf.predict_proba(X_test)
    return prediction_tree, probability_tree, tree_clf.get_depth(), tree_clf.get_n_leaves()

def bagging(X_train, X_test, y_train, y_test, n=100, m=None):
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=m), n_estimators=n, bootstrap=True)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    y_probas = bag_clf.predict_proba(X_test)
    return y_pred, y_probas

def randomForest(X_train, X_test, y_train, y_test, n=100, m=None):
    clf = RandomForestClassifier(n_estimators=n, max_depth=m)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    feature_importance = pd.Series(clf.feature_importances_, index=attribute_names).sort_values(ascending=False)
    return y_pred, y_proba, feature_importance

def adaboost(X_train, X_test, y_train, y_test, n=50, eta=1.0, m=1):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m), n_estimators=n, learning_rate=eta)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba

def xgboost(X_train, X_test, y_train, y_test, n=100, eta=0.3, m=3):
    clf = xgb.XGBClassifier(n_estimators=n, learning_rate=eta, max_depth=m, verbosity=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba


def tree(X_train, X_test, y_train, y_test):
    '''
    Model to perfrom analysis of a single tree
    Remove comments to print training score
    Plots confusion matrix, precision_recall, roc, bias, var
    '''
    treemodel = DecisionTreeClassifier()
    y_pred, y_proba, depth, leaves = decisonTree(treemodel, X_train, X_test, y_train, y_test)
    print("Accuracy", accuracy_score(y_test, y_pred))
    print("Depth = {}, n_leaves = {}".format(depth, leaves))

    depths = [1,2,3,4,5,6,7,8,9,10]
    best_depth, best_score = find_best("Decisiontree", depths)
    print("best depth = {}".format(best_depth))

    #Pruning the tree
    alphas = np.linspace(0,0.1,100)
    best_alpha, best_score = find_best("Decisiontree", alphas, 'a')
    print("Best alpha = {}".format(best_alpha))

    #Plot/print results
    treemodel = DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=best_alpha)
    y_pred, y_proba, depth, leaves = decisonTree(treemodel, X_train, X_test, y_train, y_test)

    print("Optimized tree")
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("With cvd: {}".format(np.mean(cross_val_score(treemodel, body, penguins, cv=10))))
    #treemodel.fit(X_train, y_train); training_pred = treemodel.predict(X_train)
    #print("Training data score: {}".format(accuracy_score(y_train, training_pred)))

    plots(y_test, y_pred, y_proba)

    bias, var = bias_var("Decisiontree", X_train, X_test, y_train, y_test, n_rounds=100, best_estimator=best_depth,m_depth=10)
    print("bias = {}, var = {}".format(bias, var))

def forest(X_train, X_test, y_train, y_test):
    '''
    Function for performing analysis of bagging and random forest
    The first set of plots are for random forest, second is bagging
    Remove coments to print training predictions
    '''
    results = randomForest(X_train, X_test, y_train, y_test)
    y_pred, y_proba, feature_importance = results
    print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))
    print("Feature importance: \n", feature_importance)

    sns.barplot(x=feature_importance, y=attribute_names)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Feature importance")
    plt.legend()
    plt.show()

    acc_cvd = np.mean(cross_val_score(RandomForestClassifier(n_estimators=100), body, penguins, cv=10))
    print("Cross validated score: {}".format(acc_cvd))

    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_depth, best_score = find_best("Randomforest", depths)
    print("best depth = {}".format(best_depth))

    rForest = randomForest(X_train, X_test, y_train, y_test, n=100, m=best_depth)
    print("Random forest score (w/best depth): {}".format(accuracy_score(y_test, rForest[0])))
    print("With cross-validation: {}".format(np.mean(cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=best_depth), body, penguins, cv=10))))
    rforestModel = RandomForestClassifier(n_estimators=100, max_depth=best_depth).fit(X_train, y_train); training_pred = rforestModel.predict(X_train)
    print("Training data score: {}".format(accuracy_score(y_train, training_pred)))

    bagForest = bagging(X_train, X_test, y_train, y_test, n=100, m=best_depth)
    print("Bagging score (w/best depth): {}".format(accuracy_score(y_test, bagForest[0])))
    acc_cvd_bagging = np.mean(cross_val_score(BaggingClassifier(DecisionTreeClassifier(max_depth=best_depth), n_estimators=100), body, penguins, cv=10))
    print("Bagging score = {} (with best depth & CVD)".format(acc_cvd_bagging))
    #baggingModel = BaggingClassifier(DecisionTreeClassifier(max_depth=best_depth), n_estimators=100).fit(X_train, y_train); training_pred = baggingModel.predict(X_train)
    #print("Training data score: {}".format(accuracy_score(y_train, training_pred)))

    #Plots
    y_pred, y_proba, fi = randomForest(X_train, X_test, y_train, y_test, n=100, m=best_depth)
    plots(y_test, y_pred, y_proba) #Rforest
    plots(y_test, bagForest[0], bagForest[1]) #Bagging

    print("Bias-variance tradeoff:")

    print("Random forest")
    bias, var = bias_var("Randomforest", X_train, X_test, y_train, y_test, n_rounds=100, best_estimator=best_depth, m_depth=10)
    print("bias = {}, var = {}".format(bias, var))

    print("Bagging")
    bias_b, var_b = bias_var("Bagging", X_train, X_test, y_train, y_test, n_rounds=100, best_estimator=best_depth, m_depth=10)
    print("bias = {}, var = {}".format(bias_b, var_b))

def boosting(X_train, X_test, y_train, y_test):
    '''
    Analysis of adaboost and xgboost
    comment in gridSearch() and tradeoff() to do these
    Comment in to print training data predictions
    Plots results first for adaboost, then xgBoost
    Can ignore warnings during run
    '''

    def gridSearch():
        '''
        Do gridsearch for n_estimators and learning rate, takes some time
        '''
        param_grid = {'n_estimators': [10,20,30,40,50,60,70,80,90,100,110, 120, 130, 140, 150]
                                        , 'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid)
        grid_xg = GridSearchCV(xgb.XGBClassifier(verbosity=0), param_grid)
        grid_ada.fit(X_train, y_train)
        grid_xg.fit(X_train, y_train)

        print("Adaboost")
        print("Best Params \n", grid_ada.best_params_)
        print("Score from best params: ", grid_ada.best_score_)
        print("XGboost")
        print("Best Params \n", grid_xg.best_params_)
        print("Score from best params: ", grid_xg.best_score_)

    def tradeoff():
        '''
        Perfoms bias-var tradeoff, takes some time.
        '''
        bias_ada, var_ada = bias_var("Ada", X_train, X_test, y_train, y_test, n_rounds=100, best_estimator=2, m_depth=10)
        bias_xgb, var_xgb = bias_var("XGB", X_train, X_test, y_train, y_test, n_rounds=100, best_estimator=2, m_depth=10)
        print("AdaBoost")
        print("Bias = {}, var = {}".format(bias_ada, var_ada))
        print("XGBBoost")
        print("Bias = {}, var = {}".format(bias_xgb, var_xgb))

    #gridSearch()
    #tradeoff()

    results_ada = adaboost(X_train, X_test, y_train, y_test)
    results_xg = xgboost(X_train, X_test, y_train, y_test)
    results_ada_optimized = adaboost(X_train, X_test, y_train, y_test, n=50, eta=0.5)
    results_xg_optimized = xgboost(X_train, X_test, y_train, y_test, n=100, eta=0.05)

    #Cross validation of default and optimized values
    acc_cvd_ada = np.mean(cross_val_score(AdaBoostClassifier(), body, penguins, cv=10))
    acc_cvd_xgb = np.mean(cross_val_score(xgb.XGBClassifier(verbosity=0), body, penguins, cv=10))
    acc_cvd_ada_opt = np.mean(cross_val_score(AdaBoostClassifier(n_estimators=50, learning_rate=0.5), body, penguins, cv=10))
    acc_cvd_xgb_opt = np.mean(cross_val_score(xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, verbosity=0), body, penguins, cv=10))

    #Print training predictions
    #train_clf_ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.5).fit(X_train, y_train)
    #train_clf_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, verbosity=0).fit(X_train, y_train)
    #train_pred_ada = train_clf_ada.predict(X_train)
    #train_pred_xgb = train_clf_xgb.predict(X_train)
    #print("Training ada: {}".format(accuracy_score(y_train, train_pred_ada)))
    #print("Training xgb: {}".format(accuracy_score(y_train, train_pred_xgb)))

    print("Accuracy score")
    print("Default models")
    print("AdaBoost = {}".format(accuracy_score(y_test, results_ada[0])))
    print("xgBoost = {}".format(accuracy_score(y_test, results_xg[0])))
    print("Optimized models")
    print("AdaBoost = {}".format(accuracy_score(y_test, results_ada_optimized[0])))
    print("xgBoost = {}".format(accuracy_score(y_test, results_xg_optimized[0])))

    print("Cross-validation")
    print("Default modes")
    print("Ada: {}".format(acc_cvd_ada))
    print("XGB: {}".format(acc_cvd_xgb))
    print("optimized models")
    print("Ada: {}".format(acc_cvd_ada_opt))
    print("XGB: {}".format(acc_cvd_xgb_opt))

    plots(y_test, results_ada_optimized[0], results_ada_optimized[1]) #AdaBoost
    plots(y_test, results_xg_optimized[0], results_xg_optimized[1]) #XGBoost

np.random.seed(2021)
data = load_penguins()
#dfi.export(data.head(), "penguin_describe.png")

print("Overview")
print("Features: \n", data.columns, "\n")
print("Header")
print(data.head(), "\n")
print("Class count: \n", data.groupby('species').count(), "\n")

input = data.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "species"]]
print("Features + classes: \n", input)

print("Check for absent values")
print(input.isna().sum())
print(input.shape)

print("New sample")
input.dropna(inplace=True)
input = input.reset_index(drop=True)
print(input.isna().sum())
print(input.shape)
print(input.head())

print("Class count: \n", input.groupby('species').count(), "\n")

body = input.iloc[:, :-1]
penguins = input.iloc[:, -1]
#dfi.export(body.head(), "penguins.png")

attribute_names=['bill_length', 'bill_depth', 'flipper_length', 'body_mass']
target_names=['Adelie', 'Chinstrap', 'Gentoo']

X_train, X_test, y_train, y_test = train_test(body, penguins, out=True)

'''
Comment in to do either tree, forest or boosting analysis
'''
#tree(X_train, X_test, y_train, y_test)
#forest(X_train, X_test, y_train, y_test)
#boosting(X_train, X_test, y_train, y_test)
