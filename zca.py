# -*- coding: utf-8 -*-
"""
Created on 11/05/17

@author: Thomas Pellegrini

cf https://github.com/mwv/zca/blob/master/zca/zca.py
"""

import numpy as np

class ZCA:

    def __init__(self,
                 epsilon,
                 doZCA):
        self.doZCA = doZCA
        self.epsilon = epsilon
        self.T = None
        self.invT = None


    def fit(self, X):
        """
        compute [PZ]CA transform matrix and inverse transform matrix for X
        :param X: 2-d feature matrix: nb_samples x dim_features
        :return:
        first argument: transform matrix
        second argument: inverse transform matrix
        """
        X_ = X - np.mean(X, axis=0)
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)
        U, D, _ = np.linalg.svd(cov, full_matrices=True, compute_uv=True)

        # D.clip(1e-6)
        if self.doZCA:
            print 'INFO: ZCA whitening'
            self.T = np.dot(np.dot(U, np.diag(1./np.sqrt(D + self.epsilon))), U.T)
            self.invT = np.dot(np.dot(U, np.diag(np.sqrt(D + self.epsilon))), U.T)
        else:
            # PCA whitening
            print 'INFO: PCA whitening'
            self.T = np.dot(np.diag(1. / np.sqrt(D + self.epsilon)), U.T)
            self.invT = np.dot(U, np.diag(np.sqrt(D + self.epsilon)))
        return self.T, self.invT

    def transform(self, X):
        X_ = X - np.mean(X, axis=0)
        return np.dot(X_, self.T)

    def inverse_transform(self, X, X_mean):
        return np.dot(X, self.invT) + X_mean


def standardize(X):
    """
    0-mean and 1-variance 
    :param X: 
    :return: 
    """
    return (X-np.mean(X, axis=0))/np.sqrt(np.var(X, axis=0)+1e-10)





