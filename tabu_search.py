"""Tabu search"""
"""Termination criteria: deg of dependency and feature count as fitness components, max iteration = 100"""
"""Tabu solution representation: bit string"""
"""Tabu list length: 20"""

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

def tabu_search(feature, decision, max_itr = 100):

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
    def fitness(chosen, decision, total_feature):
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
        
        return float(count) / decision.size * 80.0 + float(total_feature - chosen.shape[1]) / total_feature * 20.0

    # Generate neighborhood solution
    # Change feature combinations 90% of the time, amount of changed features is normal randomized 
    # Add or remove features happens 10% of the time
    # Modify 'solution' parameter in place
    def neighborhood_sol(solution):
        if rnd.random() < 0.90:
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

    # Improve solution and return randomly one of the best neighbors if they have same fitness
    # Use tabu list to keep last tabu_list_sz solutions
    # Regenerate new solutions using N(x) if x already exists in tabu_list
    def improve(solution, fitn, feature, decision, feature_count, tabu_list):
        best_fitn = fitn
        max_tries = round(feature_count / 2)
        tabu_list_sz = 20
        tries = 0
        sol_stack = np.stack([np.copy(solution)], axis = 0)

        while tries < max_tries:
            x = neighborhood_sol(np.copy(solution))  
            foundidx = np.where( (tabu_list == (x)).all(axis = 1) )[0] 
            while foundidx.size > 0:
                x = neighborhood_sol(np.copy(solution))  
                foundidx = np.where( (tabu_list == (x)).all(axis = 1) )[0] 

            if np.where(tabu_list != 0.)[0] > tabu_list_sz: 
                tabu_list = np.delete(tabu_list, 0, axis = 0)
            tabu_list = np.append(tabu_list, [x], axis = 0)

            new_fitn = fitness(feature[:, x], decision, feature_count)
            if new_fitn > best_fitn:
                sol_stack = np.stack([np.copy(x)], axis = 0)
                best_fitn = new_fitn
            elif new_fitn == best_fitn:
                sol_stack = np.append(sol_stack, [np.copy(x)], axis = 0)
            tries += 1        

        sol_chosen = sol_stack[rnd.randint(0, sol_stack.shape[0] - 1), :] 

        return sol_chosen
        
    tmp_sol = np.random.choice([False, True], size=feature.shape[1], p=[1./5, 4./5])
    # tmp_sol = np.random.randint(2, size=feature.shape[1]).astype(bool)
    best_sol = tmp_sol
    best_fitn = 0.0
    eval_count = 0
    tabu_list_sz = 20
    itr = 0
    plt_fitness = np.zeros(max_itr)
    tabu_list = np.zeros([tabu_list_sz, feature.shape[1]])
    while itr < max_itr:
        # improve the current solution in existing neighborhood
        tmp_sol = improve(best_sol, best_fitn, feature, decision, feature.shape[1], tabu_list)
        #tmp_sol = neighborhood_sol(np.copy(best_sol))
        print tabu_list

        # mutation operator
        for i in xrange(len(tmp_sol)):
            if rnd.random() < 0.05:
                tmp_sol[i] = not tmp_sol[i]

        # repair solution if selected feature count is zero
        if True not in tmp_sol:
            pos = rnd.randint(0, len(tmp_sol) - 1)
            tmp_sol[pos] = True

        tmp_fitn = fitness(feature[:, tmp_sol], decision, feature.shape[1]) 
        eval_count += 1
        # print itr, len(np.where(tmp_sol == True)[0]), tmp_fitn, best_fitn

        if tmp_fitn >= best_fitn:            
            best_sol = tmp_sol
            best_fitn = tmp_fitn
    
        plt_fitness[itr] = best_fitn
        itr += 1

    print len(np.where(best_sol == True)[0]), best_fitn
    return [ feature[:, best_sol], plt_fitness]

