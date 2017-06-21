# -*- coding: utf-8 -*-
"""
Created on 01/06/17

@author: Thomas Pellegrini
"""

import numpy as np
import zca

import matplotlib.pyplot as plt


# def plot_all(points, zca_projections, pca_projections, pca_vectors):
def plot_all(points, zca_projections, pca_projections):

    indices_red = np.where(points[:, 0] > 5)
    print indices_red

    # Three subplots sharing both x/y axes
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    ax1.scatter(points[:, 0], points[:, 1])
    ax1.scatter(points[indices_red, 0], points[indices_red, 1], color='r')
    ax1.set_title('original')

    ax2.scatter(pca_projections[:, 0], pca_projections[:, 1])
    ax2.scatter(pca_projections[indices_red, 0], pca_projections[indices_red, 1], color='r')
    # ax2.plot([0, 3*pca_vectors[0, 0]], [0, 3*pca_vectors[0, 1]], 'k')
    # ax2.plot([0, 3*pca_vectors[1, 0]], [0, 3*pca_vectors[1, 1]], 'k')

    # print '0:', pca_vectors[0]
    # print '1:', pca_vectors[1]

    ax2.set_title('PCA')

    ax3.scatter(zca_projections[:, 0], zca_projections[:, 1])
    ax3.scatter(zca_projections[indices_red, 0], zca_projections[indices_red, 1], color='r')
    ax3.set_title('ZCA')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=.3)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.show()
    # plt.savefig('comparison_pca_zca.png')

def generate_2d_points(nb_points, mu, sigma):
    return np.random.multivariate_normal(mu, sigma, nb_points)

def plot_2d_points(points, set_range=False):

    plt.figure()
    plt.plot(points[:, 0], points[:, 1], 'o')
    if set_range:
        plt.xlim([-3, 8])
        plt.ylim([-3, 8])
    plt.show()
    # plt.savefig('nuage.png')

nb_points=100
mu=np.array([3, 2])
sigma=np.array([[3, 1], [1, 1]])
print sigma.shape

# points = generate_2d_points(nb_points, mu, sigma)
# print points.shape
# np.save('points.npy', points)

points = np.load('points.npy')
print points.shape

# plot_2d_points(points, set_range=True)

# ZCA
transform, inv_transform = zca.whitening_fit(points, epsilon=0.0)

zca_projections = zca.transform(points, transform)

print zca_projections.shape

print np.mean(zca_projections, axis=0)
print np.cov(zca_projections.T)

# back = zca.inverse_transform(zca_projections, np.mean(points, axis=0), inv_transform)


# PCA
transform, inv_transform = zca.whitening_fit(points, epsilon=0.0, doZCA=False)

pca_projections = zca.transform(points, transform)

# print pca_projections.shape

# print np.mean(pca_projections, axis=0)
# print np.cov(pca_projections.T)

# plot_all(points, zca_projections, pca_projections)

from sklearn.decomposition import PCA

pca = PCA(n_components=2, svd_solver='full', whiten=True)
pca.fit(points)

print 'compos:', pca.components_
print 'var explained:', pca.explained_variance_ratio_

pca_projections_scikit = pca.transform(points)
# print np.cov(pca_projections_scikit.T)

plot_all(points, zca_projections, pca_projections_scikit)


# [ -1.15463195e-16  -6.88338275e-17]
# [[  1.00000000e+00   4.35515109e-16]
#  [  4.35515109e-16   1.00000000e+00]]
# (100, 2)
# [  1.42108547e-16   4.44089210e-18]
# [[ 1.44943755 -0.75456565]
#  [-0.75456565  1.08274366]]




