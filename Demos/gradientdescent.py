# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:33:47 2020

@author: BrownPlanning
"""

import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. pyplot as plt

random.seed(10)

def coef_tbl(model, X, Y):
    """X is training data used to build the model
    Y is the training data used to build the model
    """
    t = lambda M: [[row[i] for row in M] for i in range(len(M[0]))]
    b_hat = list(model.coef_)
    y_hat = model.predict(X)
    XtX = np.dot(t(X), X)
    n_ = len(X)
    mse = (sum((Y - y_hat)**2))/(n_- len(X[0]))
    var_b = mse*(np.linalg.inv(XtX).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = b_hat / sd_b
    t_val = lambda i: stats.t.cdf(np.abs(i), (n_-1))
    p_values = [2*(1 - t_val(i)) for i in ts_b]
    
    df = pd.DataFrame()
    df["Coefficients"] = b_hat
    df["Standard Errors"]= sd_b
    df["t values"] = ts_b
    df["Probabilities"] = p_values
    return df

X = [[random.random() for i in range(2)] for j in range(100)]  # x1, x2
[row.append(row[1]**2) for row in X]  # add squared term x3
[row.insert(0, 1) for row in X]  # add intercept term
X = [[round(elm, 3) for elm in row] for row in X]  # [x0,x1,x2,x3]

B = [random.random() for i in range(4)]
_sum = sum(B)
B = [round(elm/_sum, 3) for elm in B]  # standardize weights

Y = [sum(a * b for a, b in zip(B, row)) for row in X]  # Y ~ B*X

# Create linear regression object
lm = linear_model.LinearRegression(fit_intercept=False)
lm.fit(X, Y)
lm_y_hat = lm.predict(X)
b_hat = [round(i, 3) for i in list(lm.coef_)]
lmtbl = coef_tbl(lm, X, Y)
print(lmtbl)
print("Mean squared error: %.2f" % mean_squared_error(Y, lm_y_hat))
print('Variance score: %.2f' % r2_score(Y, lm_y_hat))


#Stochastic Gradient Descent
regr = linear_model.SGDRegressor(fit_intercept=False)

def get_batch(X, Y, size):
    n_ = len(X)
    subset = random.sample(range(n_), size)
    x = [X[i] for i in subset]
    y = [Y[i] for i in subset]
    return x, y

x_sets = {}
y_sets = {}
error = {}
for i in range(100):
    x_sets[i], y_sets[i] = get_batch(X, Y, 10)
    x, y = x_sets[i], y_sets[i]
    regr.partial_fit(x, y)
    error[i] = mean_squared_error(y, regr.predict(x))

sgd_y_hat = regr.predict(X)
tbl = coef_tbl(regr, x, y)
print(tbl)
print("Mean squared error: %.2f" % mean_squared_error(Y, sgd_y_hat))
print('Variance score: %.2f' % r2_score(Y, sgd_y_hat))


# # Random Forest
# from sklearn import tree
# # Make a decision tree and train
# clf = tree.DecisionTreeRegressor()
# clf = clf.fit(X, Y)
# y_pred = clf.predict(X)
# clf.score(X, Y)  # R Square
# clf.feature_importances_

# # Tree structure
# n_nodes = clf.tree_.node_count
# children_left = clf.tree_.children_left
# children_right = clf.tree_.children_right
# feature = clf.tree_.feature
# threshold = clf.tree_.threshold

# def find_path(node_numb, path, x):
#         path.append(node_numb)
#         if node_numb == x:
#             return True
#         left = False
#         right = False
#         if (children_left[node_numb] !=-1):
#             left = find_path(children_left[node_numb], path, x)
#         if (children_right[node_numb] !=-1):
#             right = find_path(children_right[node_numb], path, x)
#         if left or right :
#             return True
#         path.remove(node_numb)
#         return False
    
# def get_rule(path, column_names):
#     mask = ''
#     for index, node in enumerate(path):
#         #We check if we are not in the leaf
#         if index!=len(path)-1:
#             # Do we go under or over the threshold ?
#             if (children_left[node] == path[index+1]):
#                 mask += "(df['{}']<= {}) \t ".format(column_names[feature[node]], threshold[node])
#             else:
#                 mask += "(df['{}']> {}) \t ".format(column_names[feature[node]], threshold[node])
#     # We insert the & at the right places
#     mask = mask.replace("\t", "&", mask.count("\t") - 1)
#     mask = mask.replace("\t", "")
#     return mask

# # Leaves
# X_test = X[50:]
# leave_id = clf.apply(X_test)

# paths = {}
# for leaf in np.unique(leave_id):
#     path_leaf = []
#     find_path(0, path_leaf, leaf)
#     paths[leaf] = np.unique(np.sort(path_leaf))

# # Rules for arriving at each leaf
# rules = {}
# columns = ['Intercept', 'x0', 'x1', 'x2']
# for key in paths:
#     rules[key] = get_rule(paths[key], columns)

# values = clf.tree_.value
# # Get Leaf Predictions for each rule
# estimates = {rule: values[rule][0][0] for rule in rules}


# # Fitting Random Forest Regression to the dataset 
# # import the regressor 
# from sklearn.ensemble import RandomForestRegressor 
#  # create regressor object 
# regressor = RandomForestRegressor(n_estimators = 100, warm_start=True) 
# # fit the regressor with x and y data 
# regressor.fit(X, Y)
# regressor.feature_importances_
