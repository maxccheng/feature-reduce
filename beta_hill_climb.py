"""Beta hill climbing"""
"""Termination criteria: degree of dependency == 1.0 with max iteration = 100"""
"""Mutation characteristics: 0.5% for every bit in solution string"""

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

def beta_hill_climb(feature, decision, max_itr = 100):

    # Select unique rows from numpy array
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
        
        # Example: If fitness is 40000.11 
        #          (100000 - 40000.0) / 100000 = 0.6 which is degree of dependency 
        #          0.11 * 100 = 11 which is number of features
        return int(( 1.0 - (float(count) / decision.size) ) * 100000.0) + (chosen.shape[1] / 100.0)

    # Generate neighborhood solution with minimal changes
    # Visualize solution bit swapping as lateral movement to different slope position with same height
    #   Bit count modification as going up or down a slope 
    #   Modify solution parameter in place
    def neighborhood_sol(solution):
        if rnd.random < 0.90:
            # swap selected features
            if True in solution and False in solution:
                # swap a normalized random amount of feature pairs
                cnt = int(abs(round(np.random.normal(0, 0.5, 1)[0]))) + 1
                for k in xrange(cnt):
                    pos_a = rnd.choice(np.where(solution == True)[0])        
                    pos_b = rnd.choice(np.where(solution == False)[0])        
                    solution[pos_a] = not solution[pos_a]
                    solution[pos_b] = not solution[pos_b]
        else:
            # add or remove feature
            pos = rnd.randint(0, len(solution) - 1)        
            solution[pos] = not solution[pos]

        return solution

    def improve(solution, fitn, feature, decision):
        new_fitn = fitn
        max_tries = 10
        tries = 0
        original = np.copy(solution)

        while tries < max_tries:
            x = neighborhood_sol(solution)  
            new_fitn = fitness(feature[:, x], decision)
            if new_fitn <= fitn:
                return x
            tries += 1        

        return original
        
    tmp_sol = np.random.choice([False, True], size=feature.shape[1], p=[4./5, 1./5])
    best_sol = tmp_sol
    best_fitn = sys.float_info.max
    eval_count = 0
    itr = 0
    while itr < max_itr:
        # improve the current solution in existing neighborhood
        tmp_sol = improve(np.copy(best_sol), best_fitn, feature, decision)

        # mutate solution bit string
        for i in xrange(len(tmp_sol)):
            if rnd.random() < 0.05:
                tmp_sol[i] = not tmp_sol[i]

        # repair solution if selected feature count is zero
        if True not in tmp_sol:
            pos = rnd.randint(0, len(tmp_sol)-1)
            tmp_sol[pos] = True

        tmp_fitn = fitness(feature[:, tmp_sol], decision) 
        eval_count += 1

        if tmp_fitn <= best_fitn:            
            best_sol = tmp_sol
            best_fitn = tmp_fitn
    
        itr += 1
        # print itr, best_fitn

    print len(np.where(best_sol == True)[0]), best_fitn
    return [ feature[:, best_sol], eval_count ]

