"""Feature selection on 11 UCI datasets to minimize feature count and maximize scores"""
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

import hill_climb as hc
import beta_hill_climb as bhc
import tabu_search as ts
import simulated_annealing as sa

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

def plot_graph(xdata, ydata, fname, title, xlabel, ylabel):
    fig = plt.figure()
    plt.plot(xdata, ydata)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(fname)
    plt.close(fig)

dsets_path = "./datasets/"
dset_ext = "lung.dat"
for i, f in enumerate(sorted(os.listdir(dsets_path))):
    if f.endswith(dset_ext): 
        ds = [line.rstrip('\n').split(' ') for line in open(dsets_path + f)]
        ds = [map(int, line) for line in ds]
        # split into feature X and decision y for each instances
        X = [item[:-1] for item in ds]
        y = [item[-1] for item in ds]

        X_all_count = len(X[0])

        fselect_rep = 20
        fselect_metric = np.empty(fselect_rep, dtype='int64')
        for m in xrange(fselect_rep):
            # do feature selection
            #fselect_result = hc.hill_climb(X, y)   
            fselect_result = bhc.beta_hill_climb(X, y)   
            #fselect_result = ts.tabu_search(X, y)   
            #fselect_result = sa.simulated_annealing(X, y)   
            X_subset = fselect_result[0]  
            fselect_metric[m] = len(X_subset[0])
            search_progress = fselect_result[1]
            fname = f + "_" + str(m) + '.png'
            #plot_graph(np.arange(search_progress.size-1), search_progress[1:], fname, 'Search progress for ' + f + ' attempt #' + str(m+1), 'Generation', 'Fitness') 

            metrics = []
            train_rep = 10
            for j in xrange(train_rep):
                # hold out split with ratio 80 training : 20 test, repeated randomly for 10 times
                X_train, X_test, y_train, y_test = \
                    train_test_split(X_subset, y, test_size=0.20) 

                # train classifier
                clf = DecisionTreeClassifier()
                clf.fit(X_train, y_train) 

                # predict with test sets and save scores
                y_true = y_test
                y_predict = clf.predict(X_test)
                score = precision_recall_fscore_support(y_true, y_predict, average='macro')
                score = score[:-1]
                metrics = np.append(metrics, score) 

        # process 20 feature select reps score  
        feat_bincount = np.bincount(fselect_metric) 
        #print metrics
        pre_mean = np.mean(metrics)
        pre_min = np.min(metrics)
        pre_max = np.max(metrics)
        pre_stddev = np.std(metrics)
        print pre_mean, pre_min, pre_max, pre_stddev
        
        # print scores
        print "%02d %-15s dims=%02d trainset=%5s/%5s    feature_distribution =" % (i, f, X_all_count, len(X_train), len(X))
        feat_bincount = np.column_stack((np.arange(len(feat_bincount)), feat_bincount))
        print feat_bincount[np.where(feat_bincount[:,1] > 0)[0]]

