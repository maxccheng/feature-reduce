"""Execute wrapper method feature selection on 12 UCI datasets with DecisionTreeClassifier"""
print(__doc__)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def feature_select(feature, decision):
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
        X = feature_select(X, y)   

        # split into training and test sets randomly with defined size
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.20, random_state=20) 

        # train classifier
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train) 
                
        # verify with test sets 
        score = clf.score(X_test, y_test)        

        # calculate classification score
        print "%02d %s train_set_size=%s/%s score=%.2f" % (i, f, X_train.shape[0], X.shape[0], score)

#h = .02  # step size in the mesh
#
#names = ["DT derm", "DT exactly"]
#
#classifiers = [
#    DecisionTreeClassifier(max_depth=5),
#    DecisionTreeClassifier(max_depth=5)]
#
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
#linearly_separable = (X, y)
#
#derm = np.loadtxt("./datasets/derm.data")
#exactly = np.loadtxt("./datasets/exactly.dat")
#
#datasets = [make_moons(noise=0.3, random_state=0),
#            make_circles(noise=0.2, factor=0.5, random_state=1),
#            linearly_separable
#            ]
#
#figure = plt.figure(figsize=(27, 9))
#i = 1
## iterate over datasets
#for ds_cnt, ds in enumerate(datasets):
#    # preprocess dataset, split into training and test part
#    X, y = ds
#    X = StandardScaler().fit_transform(X)
#    X_train, X_test, y_train, y_test = \
#        train_test_split(X, y, test_size=.4, random_state=42)
#
#    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#
#    # just plot the dataset first
#    cm = plt.cm.RdBu
#    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#    if ds_cnt == 0:
#        ax.set_title("Input data")
#    # Plot the training points
#    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#    # and testing points
#    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
#    ax.set_xlim(xx.min(), xx.max())
#    ax.set_ylim(yy.min(), yy.max())
#    ax.set_xticks(())
#    ax.set_yticks(())
#    i += 1
#
#    # iterate over classifiers
#    for name, clf in zip(names, classifiers):
#        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
#        clf.fit(X_train, y_train)
#        score = clf.score(X_test, y_test)
#
#        # Plot the decision boundary. For that, we will assign a color to each
#        # point in the mesh [x_min, x_max]x[y_min, y_max].
#        if hasattr(clf, "decision_function"):
#            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#        else:
#            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#        # Put the result into a color plot
#        Z = Z.reshape(xx.shape)
#        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
#
#        # Plot also the training points
#        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#        # and testing points
#        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#                   alpha=0.6)
#
#        ax.set_xlim(xx.min(), xx.max())
#        ax.set_ylim(yy.min(), yy.max())
#        ax.set_xticks(())
#        ax.set_yticks(())
#        if ds_cnt == 0:
#            ax.set_title(name)
#        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#                size=15, horizontalalignment='right')
#        i += 1
#
#plt.tight_layout()
#plt.show()
