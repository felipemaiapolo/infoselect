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
            
def MI_gmm_reg(X, y, gmm, feat, eps=10**-50):
    
    '''
    This function calculates the mutual information between y and X in cases where we assume y continuous/with many values.
    
    Inputs: - X: numpy array of features; 
            - y: numpy array of labels;
            - gmm: GMM trained model;
            - feat: features indexes (feat);
            - eps: small value so we can avoid taking log of zero in some cases
    
    Output: - dictionary containing the estimate for the mutual information between y and X,
              and the standard deviation of measurements calculated from the samples.
    '''
    
    n, d = X.shape
    components=gmm.n_components
    Z=np.hstack((y.reshape((-1,1)),X))
    feat2=[0]+[f+1 for f in feat] #feat2 includes y in addition de X[:,feat]. PS: we sum 1 because when we train GMM, the first variable is always y

    ### Calculating log-likelihood with samples (x_i,y_i) 
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,feat2], gmm.means_[c][feat2], gmm.covariances_[c][feat2][:,feat2])

    log_like_xy=np.log(like + eps)

    
    ### Calculating log-likelihood with samples (x_i) 
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,feat2[1:]], gmm.means_[c][feat2[1:]], gmm.covariances_[c][feat2[1:]][:,feat2[1:]])

    log_like_x=np.log(like + eps)

    
    ### Calculating log-likelihood with samples (y_i) 
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,0], gmm.means_[c][0], gmm.covariances_[c][0][0])

    log_like_y=np.log(like + eps)
     
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}



def MI_gmm_class(X, y, gmm, feat, eps=10**-50):
    
    '''
    This function calculates the mutual information between y and X in cases where we assume y categorical/with few values.
    
    Inputs: - X: numpy array of features; 
            - y: numpy array of labels;
            - gmm: dict. of GMM trained models;
            - feat: features indexes (feat);
            - eps: small value so we can avoid taking log of zero in some cases
    
    Output: - dictionary containing the estimate for the mutual information between y and X,
              and the standard deviation of measurements calculated from the samples.
    '''
    
    
    n,d=X.shape
    classes=list(set(y))
    p={}

    ### Calculating log-likelihood with samples (y_i) 
    like=np.zeros(n)
    for c in classes:
        p[c]=np.mean(y==c)
        like[y==c]=p[c]

    log_like_y=np.log(like + eps)
    
    
    ### Calculating log-likelihood with samples (x_i,y_i) 
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux[y==c]+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[y==c][:,feat], gmm[c].means_[comp][feat], gmm[c].covariances_[comp][feat][:,feat])

        #(X,Y)
        like[y==c]=p[c]*like_aux[y==c] 
    log_like_xy=np.log(like + eps)

    
    ### Calculating log-likelihood with samples (x_i)
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[:,feat], gmm[c].means_[comp][feat], gmm[c].covariances_[comp][feat][:,feat])

        #Marginalization of (X,Y)
        like+=p[c]*like_aux

    log_like_x=np.log(like + eps)
    
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}



def MI(cand, posic, r, X, y, gmm, include_cand = True, eps=10**-50):
    
    '''
    This function is an intermediary function between the main class and the two functions that make the calculation of the
    mutual information. It basically decides which of the two functions to use and if we should do the forward or backward step.
    
    Inputs: - cand: position of the candidate variable to be chosen;
            - posic: list with positions of the selected variables so far;
            - r: round;
            - X: numpy array of features; 
            - y: numpy array of labels;
            - gmm: model or dict. of GMM(s);
            - include_cand: include or remove variables (forwar/backward)d;
            - eps: small value so we can avoid taking log of zero in some cases
            
    Output: - cand: position of the candidate variable to be chosen;
            - dic: dictionary containing the estimate for the mutual information between y and X,
              and the standard deviation of measurements calculated from the samples. 
    '''
        
    n,d=X.shape
    aux = copy.deepcopy(posic)
    if include_cand:
        aux[r] = cand
    else:
        aux.remove(cand)
    
    if type(gmm)==dict:
        dic=MI_gmm_class(X, y, gmm, aux, eps)
    else:
        dic=MI_gmm_reg(X, y, gmm, aux, eps) 
    
    return cand, dic