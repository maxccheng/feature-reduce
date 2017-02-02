"""Execute wrapper method feature selection on 12 UCI datasets with DecisionTreeClassifier"""
print(__doc__)

# suppress warnings e.g. divide by 0 in precision_recall_fscore_support()
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

def feature_select(feature, decision, search_type = "forward_search"):

    def unique_rows(a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    def fitness(chosen, decision):
        uniques = unique_rows(chosen)        
        count = 0
        for eqclass in uniques:
            instances = uniques.find(eqclass)
            # todo: reduce to flagged instances, not all of them 
            deci = None
            classifiable = True 
            for i in instances:
                if deci == None:
                    deci = decision[i] # todo: fix this i index
                else if deci != decision[i]:
                    classifiable = False
                    break 
            if classifiable == True:
                count = count + instances.size # todo: fix this instance and size thingy

        return count / decision.size
        
    if search_type == "forward_search":
        candidates = np.arange(feature.shape[1])
        solution = np.zeros(0)
        best_fitness = 0
        while candidates.size != 0:
            # select next fittest feature to add
            nextf = -1
            for j in range(len(candidates)):
                if dependency_degree(solution, candidates[j]) > best_fitness:            
                    nextf = j
                    break

            # remove i-th feature from pool 
            x = candidates[nextf]
            solution = np.append(solution, x)  
            candidates = np.delete(candidates, [i])

            # break if reduct set found

    else if search_type == "backward_search":


    return feature

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
        if i == 1:
            X = feature_select(X, y)   

        metrics = []
        rep = 10
        for j in range(rep):
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
        print "%02d %-15s dim_selected=%02d train_instances=%5s/%5s precision=%.2f recall=%.2f fscore=%.2f" % (i, f, X.shape[1], X_train.shape[0], X.shape[0], metrics_avg[0], metrics_avg[1], metrics_avg[2])

