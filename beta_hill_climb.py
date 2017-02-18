"""Beta hill climbing and forward search"""
"""Termination criteria: degree of dependency == 1.0"""
"""Mutation chance: 5%"""
"""Mutation bit length: randomize 1 bit/feature"""

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

def beta_hill_climb(feature, decision, search_type = "forward_search", mutate_chance = 0.1):

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
        eval_count = 0
        solution = np.zeros(feature.shape[1], dtype=bool)
        fitn_best = 0.0
        while fitn_best < 1.0:
            feat_best = -1  # select next fittest feature

            for j in np.where(solution == False)[0]:
                solution_tmp = np.copy(solution)
                solution_tmp[j] = True 
                fitn_tmp = degree_dependency(feature[:,solution_tmp], decision) 
                eval_count += 1
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
                eval_count += 1

    return [ feature[:,solution], eval_count ]

