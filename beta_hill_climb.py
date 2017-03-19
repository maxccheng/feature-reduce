"""Beta hill climbing"""
"""Termination criteria: deg of dependency and feature count as fitness components, max iteration = 100"""
"""Mutation operator: 5% to 0.05% for every bit in solution"""

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

def beta_hill_climb(feature, decision, max_itr = 100):

    # Calculate degree of dependency
    # By summing all element counts of each equivalent class
    #   that can be classified without ambiguity
    def deg_of_dep(feature, solution, decision, feature_count):
        chosen = [ list(itl.compress(x, solution)) for x in feature ]
        uniques = [ list(x) for x in set(tuple(x) for x in chosen) ] # select one instance out of each equivalent class 
        count = 0
        for eqclass in uniques:
            matched_idx = [ i for i,x in enumerate(uniques) if x == eqclass ] # index list of each equivalent class
            deci = None
            classifiable = True 
            for i in matched_idx:
                if deci is None:
                    deci = decision[i] 
                elif deci != decision[i]:
                    classifiable = False
                    break 
            if classifiable == True:
                count = count + len(matched_idx)

        result = float(count) / len(feature)
        if result >= 1.00:
            result = 1.05
        return result

    def fitness(feature, solution, decision, total_feature):
        count = solution.count(True)
        if  count <= 0:
            return 0.00

        return deg_of_dep(feature, solution, decision, total_feature) * 99.0 + \
               float(total_feature - count) / total_feature * 1.0

    # Generate neighborhood solution
    # by flipping one feature state at a time
    # then return all possible trial solutions
    def neighborhood_sol(sol):
        return [ [ False if j == i else p for j,p in enumerate(sol) ] for i,x in enumerate(sol) ]

    # Improve solution and return randomly one of the best neighbors if they have same fitness
    def improve(feature, solution, fitn, decision, feature_count):
        trials = neighborhood_sol(solution)  
        fit_val = [ fitness(feature, solution, decision, feature_count) for x in trials ]
        max_idx = [ i for i,x in enumerate(fit_val) if x == max(fit_val) ]
        sol_chosen = trials[rnd.choice(max_idx)]

        return sol_chosen
       
    feat_count = len(feature[0]) 
    tmp_sol = [True] * feat_count
    tmp_fitn = 0.00
    best_sol = tmp_sol
    best_fitn = tmp_fitn
    eval_count = 0
    itr = 0
    plt_fitness = np.zeros(max_itr)
    while itr < max_itr:
        # improve the current solution in existing neighborhood
        tmp_sol = improve(feature, best_sol, best_fitn, decision, feat_count)

        # mutation operator
        tmp_fitn = fitness(feature, tmp_sol, decision, feat_count) 
        if tmp_fitn == best_fitn:
            for i in xrange(len(tmp_sol)):
                if rnd.random() < 0.05:
                    tmp_sol[i] = not tmp_sol[i]

        # repair solution if selected feature count is zero
        if True not in tmp_sol:
            pos = rnd.randint(0, len(tmp_sol) - 1)
            tmp_sol[pos] = True

        tmp_fitn = fitness(feature, tmp_sol, decision, feat_count) 
        eval_count += 1
        #print itr, len(list(itl.compress(tmp_sol, tmp_sol))), tmp_fitn, len([i for i,x in enumerate(best_sol) if x]), best_fitn

        if tmp_fitn > best_fitn:            
            best_sol = tmp_sol
            best_fitn = tmp_fitn
    
        plt_fitness[itr] = best_fitn
        itr += 1

    print len([i for i,x in enumerate(best_sol) if x]), best_fitn
    return [ [ list(itl.compress(x, best_sol)) for x in feature ], plt_fitness]

