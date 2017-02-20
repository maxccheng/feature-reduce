"""Greedy hill climbing"""
"""Termination criteria: degree of dependency == 1.0 with max iteration = 100"""

# suppress warnings e.g. divide by 0 in precision_recall_fscore_support()
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pudb 
import os
import sys
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itl
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

def greedy_hill_climb(feature, decision, max_itr = 100):

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
    def fitness(chosen, decision):
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

        return int(( 1.0 - (float(count) / decision.size) ) * 100000.0) + (chosen.shape[1] / 100.0)

    # Generate neighborhood solution with minimal changes
    # Visualize solution bit swapping as lateral movement to different slope position with same height
    #   Bit count modification as going up or down a slope 
    #   Repair solutions that have empty featureset
    def neighborhood_sol(solution):
        # different features but same count
        if True in solution and False in solution:
            pos_a = rnd.choice(np.where(solution == True)[0])        
            pos_b = rnd.choice(np.where(solution == False)[0])        
            solution[pos_a] = not solution[pos_a]
            solution[pos_b] = not solution[pos_b]

        # modify count 
        if rnd.random() < 0.05:
            pos = rnd.randint(0, len(solution) - 1)        
            solution[pos] = not solution[pos]

        # repair solution if selected feature count is zero
        if True not in solution:
            pos = rnd.randint(0, len(solution)-1)
            solution[pos] = True

        return solution
        
    tmp_sol = np.random.choice([False, True], size=feature.shape[1], p=[4./5, 1./5])
    best_sol = tmp_sol
    best_fitn = sys.float_info.max
    eval_count = 0
    itr = 0
    while itr < max_itr:

        tmp_sol = neighborhood_sol(best_sol)
        tmp_fitn = fitness(feature[:, tmp_sol], decision) 
        eval_count += 1
        if tmp_fitn <= best_fitn:            
            # print len(np.where(best_sol == True)[0]), best_fitn, tmp_fitn
            best_sol = tmp_sol
            best_fitn = tmp_fitn
    
        itr += 1

    print len(np.where(best_sol == True)[0]), best_fitn
    return [ feature[:, best_sol], eval_count ]

