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
import math
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
    def deg_of_dep(chosen, decision, total_feature):
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
        
        result = float(count) / decision.size 
        
        return result

    def fitness(chosen, decision, total_feature):
        result = deg_of_dep(chosen, decision, total_feature)
        perfect_deg = 0.0
        if result == 1.0:
            perfect_deg = 1.0

        return deg_of_dep(chosen, decision, total_feature) * 75.0 + \
               float(total_feature - chosen.shape[1]) / total_feature * 20.0 + \
               perfect_deg * 5.0

    # Generate neighborhood solution
    # Change feature combinations 90% of the time, amount of changed features is number of bits / 2
    # Add or remove features happens 10% of the time
    # Modify 'solution' parameter in place
    def neighborhood_sol(solution):
        if rnd.random() < 0.90:
            if False in solution:
                for i in xrange(len(solution) / 2):
                    pos_a = rnd.choice(np.where(solution == True)[0])        
                    pos_b = rnd.choice(np.where(solution == False)[0])        
                    solution[pos_a] = not solution[pos_a]
                    solution[pos_b] = not solution[pos_b]
        else:
            # balance add/remove feature chances to 50% addition and 50% removal
            rem_min = min(len(np.where(solution == True)[0]) - 1, 3)
            add_rem = rnd.randint(-rem_min, 2)
            if add_rem >= 0:
                add_rem += 1 # after this becomes -3,-2,-1,1,2,3 if >3 features chosen

            for i in xrange(abs(add_rem)):
                if add_rem > 0:
                    if False in solution:
                        pos = rnd.choice(np.where(solution == False)[0])        
                        solution[pos] = True
                else:
                    if True in solution:
                        pos = rnd.choice(np.where(solution == True)[0])        
                        solution[pos] = False

        return solution

    # Improve solution and return randomly one of the best neighbors if they have same fitness
    def improve(solution, fitn, feature, decision, feature_count):
        best_fitn = fitn
        multiplier = 5.0 * (1.0 - abs(len(np.where(solution)[0]) - (0.5 * feature_count)) / (0.5 * feature_count))
        max_tries = feature_count * (multiplier + 1.0)
        tries = 0
        sol_stack = np.stack([np.copy(solution)], axis = 0)

        while tries < max_tries:
            x = neighborhood_sol(np.copy(solution))  
            new_fitn = fitness(feature[:, x], decision, feature_count)
            if new_fitn > best_fitn:
                sol_stack = np.stack([np.copy(x)], axis = 0)
                best_fitn = new_fitn
            elif new_fitn == best_fitn:
                sol_stack = np.append(sol_stack, [np.copy(x)], axis = 0)
            tries += 1        

        sol_chosen = sol_stack[rnd.randint(0, sol_stack.shape[0] - 1), :] 

        return sol_chosen, max_tries
        
    tmp_sol = np.random.choice([False, True], size=feature.shape[1], p=[1./10, 9./10])
    # repair solution if selected feature count is zero
    if True not in tmp_sol:
        pos = rnd.randint(0, len(tmp_sol) - 1)
        tmp_sol[pos] = True
    best_sol = tmp_sol
    best_fitn = fitness(feature[:, tmp_sol], decision, feature.shape[1]) 
    eval_count = 1
    itr = 0
    plt_fitness = np.zeros(max_itr)
    while itr < max_itr:
        tmp_sol = np.copy(best_sol)
        # mutation operator
        for i in xrange(len(tmp_sol)):
            if rnd.random() < 0.01: 
                tmp_sol[i] = not tmp_sol[i]

        # repair solution if selected feature count is zero
        if True not in tmp_sol:
            pos = rnd.randint(0, len(tmp_sol) - 1)
            tmp_sol[pos] = True

        # improve the current solution in existing neighborhood
        tmp_sol, improve_eval = improve(best_sol, best_fitn, feature, decision, feature.shape[1])

        tmp_fitn = fitness(feature[:, tmp_sol], decision, feature.shape[1]) 
        eval_count += improve_eval + 1

        print   str(itr).rjust(3), \
                str(len(np.where(tmp_sol)[0])).rjust(3), \
                str(format(tmp_fitn,'.4f')).rjust(8), \
                str(len(np.where(best_sol)[0])).rjust(3), \
                str(format(best_fitn,'.4f')).rjust(8), \
                str(int(eval_count)).rjust(6)

        if tmp_fitn >= best_fitn: 
            best_sol = tmp_sol
            best_fitn = tmp_fitn

        plt_fitness[itr] = best_fitn
        itr += 1

    print   str(len(np.where(best_sol)[0])).rjust(2), \
            str(format(best_fitn,'.4f')).rjust(8), \
            str(int(eval_count)).rjust(6), \
            np.where(best_sol == True)[0]

    return [ feature[:, best_sol], plt_fitness]

