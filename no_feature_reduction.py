"""Control set classification scores with no feature selection on 11 UCI datasets"""
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
        #metrics = metrics.reshape(rep, 3)
        metrics_mean = np.mean(metrics, axis=0) 
        metrics_min = np.min(metrics, axis=0) 
        metrics_max = np.max(metrics, axis=0) 
        metrics_stddev = np.std(metrics, axis=0) 

        # print average scores
        print "%02d %-15s dim_selected=%02d/%02d train_instances=%5s/%5s fscore=%.2f %.2f %.2f %.2f" % (i, f, X.shape[1], X_all_count, X_train.shape[0], X.shape[0], metrics_mean, metrics_min, metrics_max, metrics_stddev)

