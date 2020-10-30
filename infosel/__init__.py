
name="info_selection/info_selection"
__version__ = "1.0.0"

###
import math
import numpy as np  
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from .gmm import *
from .mi import *

def check_array(X, name="X", dim=2): 
    if not (type(X)==np.ndarray) or len(X.shape)!=dim:
            raise ValueError(name+" should be a {:}-dimensional Numpy array.".format(dim))
    
class SelectVars:
    
    '''
    This is the main class of the package.
    '''
    
    selection_mode = None
    gmm = None
    n = None
    
    def __init__(self, gmm, selection_mode = 'forward'):
        """
        Inputs: - gmm: model or dict. of GMM(s);
                - selection_mode: forward/backward algorithms.
        """
        
        if not selection_mode in ['forward', 'backward']:
            raise ValueError("Selection model should be either 'forward' or 'backward'.")
            
        self.selection_mode = selection_mode
        self.gmm=gmm
    
    def fit(self, X, y, verbose=True, eps=10**-50):
        
        '''
        This function order the features according to their importance - from 
        most important to least important (forward) or from least important to most importante (backward).

        Inputs: - X: numpy array of features; 
                - y: numpy array of labels;
                - verbose: print or not to print!?
                - eps: small value so we can avoid taking log of zero in some cases 
        '''
        
        #Checking input format
        check_array(X, name="X", dim=2)
        check_array(y, name="y", dim=1)
        
        '''Creating some important objects'''
        self.n, self.d = X.shape
        include_var = self.selection_mode == 'forward'  #True if include or False if remove

        self.delta_list = [] # list with history of % changes of mutual info when adding/removing the best/worst variables
        self.mi_list = []  # list with mutual info history by adding/removing the best/worst variables
        self.stds_list = []  # list with stds history and that we will use to calculate the standard error of MIs
        self.feat_hist=[] # history of variables at each round
        lista = list(range(self.d))  # list with indexes of all variables
        
        '''Defining number of iterations and list of features we use in each iteration'''
        if verbose: print("Let's start...\n")
            
        #The 'posic' list starts empty if you include
        if include_var:
            posic = []  # lista com posições das variáveis selecionadas até o momento
            self.feat_hist.append(copy.deepcopy(posic))
            rounds = range(self.d) 

            self.mi_list.append(0)
            self.stds_list.append(0)
            self.delta_list.append(0)
            
        #The 'posic' list starts full if we take it out
        else:
            posic = copy.deepcopy(lista)
            self.feat_hist.append(copy.deepcopy(posic))
            rounds = range(self.d-1) 

            if type(self.gmm)==dict:
                dic=MI_gmm_class(X, y, self.gmm, posic, eps)
            else:
                dic=MI_gmm_reg(X, y, self.gmm, posic, eps) 

            self.mi_list.append(dic['mi'])
            self.stds_list.append(dic['std'])
            self.delta_list.append(0)
        
        if verbose: print("Round = {:3d}   |   Î = {:5.2f}   |   Δ%Î = {:5.2f}   |   Features={}".format(0, self.mi_list[-1], 0, posic)) 
        
        
        '''Calculating the Mutual Information (forward or backward fashion)'''
        for r in rounds: # "r" of rounds/repetitions
         
            if include_var:
                posic.append(None)
                
            #Calcula MI entre y e X[:,(posic, cand)] -> cand: variável candidata a ser selecionada
            outputs = [MI(cand, posic, r, X, y, self.gmm, include_var, eps) for cand in lista]
    
            #Escolhendo variável que traz maior retorno
            MI_best=-math.inf
            
            for out in outputs:
                
                cand, dic = out
                MI_current = dic['mi']
                
                if MI_current > MI_best:
                    MI_best = MI_current
                    std_best = dic['std']
                    best_index = cand
         
            #Δ%Î
            if r==0 and include_var:
                self.delta_list.append(0)
            else:
                self.delta_list.append(MI_best/self.mi_list[-1]-1)
                
            #Updating variable list        
            lista.remove(best_index)
            if include_var:
                posic[r] = best_index
            else:
                posic.remove(best_index)
    
            #Updating lists 
            self.mi_list.append(MI_best)
            self.stds_list.append(std_best)
            self.feat_hist.append(copy.deepcopy(posic))
            
            #Verbose
            if verbose: print("Round = {:3d}   |   Î = {:5.2f}   |   Δ%Î = {:5.2f}   |   Features={}".format(r+1, MI_best, self.delta_list[-1], posic))
            

        
    def get_info(self): 
        
        '''
        This function creates and outputs a Pandas DataFrame with the history of feature importance.
        '''
        
        dic={'rounds': range(0,len(self.mi_list)), 
             'mi_mean': self.mi_list, 
             'mi_error': [s/np.sqrt(self.n) for s in self.stds_list],
             'delta': self.delta_list,
             'features':self.feat_hist,
             'num_feat':[len(l) for l in self.feat_hist]}
        return pd.DataFrame(dic).loc[:,['rounds','mi_mean','mi_error','delta','num_feat','features']]
     
    def plot_delta(self): 
        
        '''
        This function plots the history of percentual changes in the mutual information.
        '''
        
        l=self.delta_list
        plt.plot(list(range(1,len(l))),l[1:])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Rounds")
        plt.ylabel("Δ% Mutual Information")
        plt.show()
        
    def plot_mi(self): 
        
        '''
        This function plots the history of the mutual information.
        '''
        
        l,s=self.mi_list, self.stds_list
        plt.errorbar(list(range(len(l))), l, yerr=np.array(s)/np.sqrt(self.n)) #
        plt.axhline(y=0, color='g', linestyle='--')
        plt.xlabel("Rounds")
        plt.ylabel("Mutual Information")
        plt.show()
        
    def transform(self, X, rd): 
        
        '''
        This transforms X using the round 'rd'. Examine the history dataframe and plot before choosing 'rd'.
        '''
        
        return X[:,self.get_info().loc[rd,'features']]