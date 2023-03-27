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
            
def gmm_scores(X_train, X_val, k=3, covariance_type='full', reg_covar=1e-06, random_state=42):
    
    '''
    This function trains a GMM and evaluate it in a holdout set using the mean log_likelihood of samples
    
    Inputs: - X_train: training set; 
            - k: number of GMM components;
            - covariance_type: covariance type (scikit-learn implementation);
     
            - X_val: holdout set (used if criterion=="loglike"); 
            - random_state: seed.
    
    Output: - score
    '''
    
    assert covariance_type in ['full','tied','diag','spherical'] 
    
    clf = mixture.GaussianMixture(n_components=k, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state, max_iter=1000)
    clf.fit(X_train)
    return clf.score(X_val)

def edit_covariances(gmm, covariance_type, d):
    
    n_comp = gmm.n_components
    
    if covariance_type=='spherical':
        covs = np.array([(var*np.eye(d)).tolist() for var in gmm.covariances_])  
    
    elif covariance_type=='diag':
        covs = np.array([np.diag(var).tolist() for var in gmm.covariances_])
    
    elif covariance_type=='tied':
        covs = np.array(n_comp*[gmm.covariances_.tolist()])
    
    else:
        covs = gmm.covariances_
        
    gmm.covariances_ = covs
    
    return gmm

def get_gmm(X, y, y_cat=False, num_comps=[2,5,10,15,20,30,40,50], val_size=0.33, reg_covar=1e-06, covariance_type='full', random_state=42):
    
    '''
    This function trains a GMM and evaluate it in a holdout set using the mean log_likelihood of samples
    
    Inputs: - X: numpy array of features; 
            - y: numpy array of labels;
            - y_cat: if we should consider y as categorical;
            - num_comps: numbers of GMM components to be tested;
            - val_size: size of holdout set used to validate the GMMs numbers of components
            - reg_covar: covariance regularization (scikit-learn implementation);
            - covariance_type: covariance type (scikit-learn implementation);
            - random_state: seed.
    
    Output: - GMM ou dictionary of GMMs
    '''
    
    #Checking input format
    check_array(X, name="X", dim=2)
    check_array(y, name="y", dim=1)
    assert covariance_type in ['full','tied','diag','spherical'] 
    
    #Y categorical/or with few values
    if y_cat: 
        classes=list(set(y))
        gmm={}

        for c in classes:
            #Selecting number of components
            X_gmm_train, X_gmm_val, _, _=train_test_split(X[y==c], X[y==c], test_size=val_size, random_state=random_state)
            scores=np.array([gmm_scores(X_gmm_train, X_gmm_val, k, covariance_type=covariance_type, reg_covar=reg_covar,  random_state=random_state) for k in num_comps])
            k_star=num_comps[np.argmax(scores)]

            #Training GMMs
            gmm[c] = mixture.GaussianMixture(n_components=k_star, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state)
            gmm[c].fit(X[y==c])
            gmm[c] = edit_covariances(gmm[c], covariance_type, X.shape[1])
        
        return gmm #it is a dictionary of GMMs
    
    #Y continuous/or with many values
    else: 
        #Selecting number of components        
        X_gmm_train, X_gmm_val, y_gmm_train, y_gmm_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
        Z_gmm_train=np.hstack((y_gmm_train.reshape((-1,1)), X_gmm_train))
        Z_gmm_val=np.hstack((y_gmm_val.reshape((-1,1)), X_gmm_val))
        scores=np.array([gmm_scores(X_train=Z_gmm_train, X_val=Z_gmm_val, k=k, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state) for k in num_comps])
        k_star=num_comps[np.argmax(scores)]
        
        #Training GMM
        Z = np.hstack((y.reshape((-1,1)),X))
        gmm = mixture.GaussianMixture(n_components=k_star, covariance_type=covariance_type, reg_covar=reg_covar, random_state=random_state)
        gmm.fit(Z)
        gmm = edit_covariances(gmm, covariance_type, Z.shape[1])
        
        return gmm #it is a GMM