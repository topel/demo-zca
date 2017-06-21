# -*- coding: utf-8 -*-
"""
Created on 11/05/17

@author: Thomas Pellegrini

cf https://github.com/mwv/zca/blob/master/zca/zca.py
"""

import numpy as np


def standardize(X):
    """
    0-mean and 1-variance 
    :param X: 
    :return: 
    """
    return (X-np.mean(X, axis=0))/np.sqrt(np.var(X, axis=0)+1e-10)

def whitening_fit(X, epsilon=0.01, doZCA=True):
    """
    whiten X
    :param X: 2-d feature matrix: nb_samples x dim_features 
    :param eps: 
    :return: 
    first argument: transform matrix
    second argument: inverse transform matrix
    """
    X_ = X - np.mean(X, axis=0)
    cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
    U, D, _ = np.linalg.svd(cov, full_matrices=True, compute_uv=True)

    # D.clip(1e-6)
    if doZCA:
        return np.dot(np.dot(U, np.diag(1./np.sqrt(D + epsilon))), U.T), np.dot(
            np.dot(U, np.diag(np.sqrt(D+epsilon))), U.T)
    else: # PCA
        return np.dot(np.diag(1. / np.sqrt(D + epsilon)), U.T), np.dot(
            U, np.diag(np.sqrt(D + epsilon)))

def transform(X, T):
    X_ = X - np.mean(X, axis=0)
    return np.dot(X_, T)


def inverse_transform(X, X_mean, Tinv):
    return np.dot(X, Tinv) + X_mean




