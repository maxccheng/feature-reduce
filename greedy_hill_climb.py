"""Feature selection on 11 UCI datasets to minimize feature count and maximize scores"""
"""Greedy hill climbing, forward search"""
"""Termination criteria: degree of dependency == 1.0"""
print(__doc__)

# suppress warnings e.g. divide by 0 in precision_recall_fscore_support()
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pudb 
import os
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

def greedy_hill_climb(feature, decision, search_type = "forward_search"):

    # Utility function
    def unique_rows(a):
        if a.size:
            a = np.ascontiguousarray(a)
            unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
            return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
        else:
            return np.zeros(0)

    # Calculate degree of dependency
    # By summing all element counts of each equivalent class
    #   that can be classified without ambiguity
    def degree_dependency(chosen, decision):
        uniques = unique_rows(chosen)                                   # select one instance out of each equivalent class 
        count = 0
        for eqclass in uniques:
            matched_idx = np.where((chosen == (eqclass)).all(axis = 1)) # get index list of i-th equivalent class
            deci = None
            classifiable = True 
            for i in matched_idx[0]:
                if deci is None:
                    deci = decision[i] 
                elif deci != decision[i]:
                    classifiable = False
                    break 
            if classifiable == True:
                count = count + matched_idx[0].size 

        return float(count) / decision.size
        
    if search_type == "forward_search":
        candidates = np.arange(feature.shape[1])
        solution = np.empty([feature.shape[0], 0])
        best_fitness = 0.0
        while best_fitness != 1.0:
            # select next fittest feature to add
            nextf = -1
            for j in xrange(len(candidates)):
                f = degree_dependency(np.append(solution, feature[:,[j]], 1), decision) 
                if f > best_fitness:            
                    best_fitness = f
                    nextf = j

            # remove i-th feature from pool 
            solution = np.append(solution, feature[:,[nextf]], 1)  
            #print solution
            feature = np.delete(feature, nextf, 1)
            candidates = np.delete(candidates, nextf, 0)
        
    return solution

dsets_path = "./datasets/"
dset_ext = ".dat"
for i, f in enumerate(os.listdir(dsets_path)):
    if f.endswith(dset_ext): 
        ds = np.loadtxt(dsets_path + f)
        ds = ds.astype(int)
        # split into feature and decision with rows as instances
        X = ds[:, :-1] 
        y = ds[:, -1].reshape(-1, 1)

        # do feature selection
        X_all_count = X.shape[1]
        X = greedy_hill_climb(X, y)   
        #np.set_printoptions(threshold=100000)

        metrics = []
        rep = 10
        for j in xrange(rep):
            # hold out split with ratio 80 training : 20 test, repeated randomly for 10 times
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.20) 

            # train classifier
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train) 

            # predict with test sets and save scores
            y_true = y_test
            y_predict = clf.predict(X_test)
            score = precision_recall_fscore_support(y_true, y_predict, average='macro')
            score = score[:-1]
            metrics = np.append(metrics, score) 

        # average 10 repeat scores
        metrics = metrics.reshape(rep, 3)
        metrics_avg = np.mean(metrics, axis=0) 

        # print average scores
        print "%02d %-15s dim_selected=%02d/%02d train_instances=%5s/%5s precision=%.2f recall=%.2f fscore=%.2f" % (i, f, X.shape[1], X_all_count, X_train.shape[0], X.shape[0], metrics_avg[0], metrics_avg[1], metrics_avg[2])

