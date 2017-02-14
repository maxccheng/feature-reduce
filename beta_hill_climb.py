"""Feature selection on 11 UCI datasets to minimize feature count and maximize scores"""
"""Beta hill climbing and forward search"""
"""Termination criteria: degree of dependency == 1.0"""
"""Mutation chance: 5%"""
"""Mutation bit length: randomize 1 bit/feature"""
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

def beta_hill_climb(feature, decision, search_type = "forward_search", mutate_chance = 0.5):

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
        solution = np.zeros(feature.shape[1], dtype=bool)
        fitn_best = 0.0
        while fitn_best < 1.0:
            feat_best = -1  # select next fittest feature

            for j in np.where(solution == False)[0]:
                solution_tmp = np.copy(solution)
                solution_tmp[j] = True 
                fitn_tmp = degree_dependency(feature[:,solution_tmp], decision) 
                if fitn_tmp > fitn_best:            
                    fitn_best = fitn_tmp
                    feat_best = j 
        
            if feat_best < 0:
                feat_best = np.where(solution == False)[0][0]
            solution[feat_best] = True  # add the fittest feature
            
            if rnd.random() < mutate_chance:
                midx = rnd.randint(0, len(solution)-1) 
                # print "mutate idx %d" % midx
                solution[midx] = not solution[midx]    
                fitn_best = degree_dependency(feature[:,solution], decision) 

    return feature[:,solution]

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
        X = beta_hill_climb(X, y)   
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

