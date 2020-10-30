import math
import numpy as np  
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
    
def check_array(X, name="X", dim=2): 
    if not (type(X)==np.ndarray) or len(X.shape)!=dim:
            raise ValueError(name+" should be a {:}-dimensional Numpy array.".format(dim))
            
def gmm_scores(X_train, X_val, k, covariance_type='full', random_state=42):
    
    '''
    This function trains a GMM and evaluate it in a holdout set using the mean log_likelihood of samples
    
    Inputs: - X_train: training set; 
            - X_val: holdout set; 
            - k: number of GMM components;
            - covariance_type: covariance type (scikit-learn implementation);
            - random_state: seed.
    
    Output: - log_likelihood in the holdout set
    '''
    
    clf = mixture.GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=random_state)
    clf.fit(X_train)
    return clf.score(X_val)

def get_gmm(X, y, y_cat=False, covariance_type='full', max_comp=20, val_size=0.33, random_state=42):
    
    '''
    This function trains a GMM and evaluate it in a holdout set using the mean log_likelihood of samples
    
    Inputs: - X: numpy array of features; 
            - y: numpy array of labels;
            - y_cat: if we should consider y as categorical;
            - covariance_type: covariance type (scikit-learn implementation);
            - max_comp: maximum number of GMM components to be tested;
            - val_size: size of holdout set used to validate the GMMs numbers of components
            - random_state: seed.
    
    Output: - GMM ou dictionary of GMMs
    '''
    
    #Checking input format
    check_array(X, name="X", dim=2)
    check_array(y, name="y", dim=1)
    
    #Y categorical/or with few values
    if y_cat: 
        classes=list(set(y))
        gmm={}

        for c in classes:
            #Selecting number of components
            X_gmm_train, X_gmm_val, _, _=train_test_split(X[y==c], X[y==c], test_size=val_size, random_state=random_state)
            scores=np.array([gmm_scores(X_gmm_train, X_gmm_val, k, covariance_type=covariance_type, random_state=random_state) for k in list(range(1, max_comp, 1))])
            k_star=np.argmax(scores)+1

            #Training GMMs
            gmm[c] = mixture.GaussianMixture(n_components=k_star, covariance_type=covariance_type, random_state=random_state)
            gmm[c].fit(X[y==c])
        
        return gmm #it is a dictionary of GMMs
    
    #Y continuous/or with many values
    else: 
        #Selecting number of components
        X_gmm_train, X_gmm_val, y_gmm_train, y_gmm_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
        Z_gmm_train=np.hstack((y_gmm_train.reshape((-1,1)), X_gmm_train))
        Z_gmm_val=np.hstack((y_gmm_val.reshape((-1,1)), X_gmm_val))
        scores=np.array([gmm_scores(Z_gmm_train, Z_gmm_val, k, covariance_type=covariance_type, random_state=random_state) for k in list(range(1, max_comp, 1))])
        k_star=np.argmax(scores)+1
        
        #Training GMM
        Z = np.hstack((y.reshape((-1,1)),X))
        gmm = mixture.GaussianMixture(n_components=k_star, covariance_type=covariance_type, random_state=random_state)
        gmm.fit(Z)
        
        return gmm #it is a GMM