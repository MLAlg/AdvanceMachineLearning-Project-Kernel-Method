#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:19:22 2019
@author: Samaneh
"""

from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from AML.Utils.kernels import Kernels
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from AML.dataset import load_hoda


class kPCA(Kernels):
    def __init__(self, k = None, kernel = None):
        super().__init__()
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        if not kernel:
            kernel = 'rbf'
            self.kernel = kernel
        else:
            self.kernel = kernel
        return
    
    def explained_variance_(self):
        '''
        :Return: explained variance.
        '''
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse = True)[:self.k]]
        return self.explained_variance
    
    def kernelize(self, x1, x2):
        '''
        :params: x1: NxD
        :params: x2: NxD
        '''
        if self.kernel == 'linear':
            return Kernels.linear(x1, x2)
        elif self.kernel == 'rbf':
            return Kernels.rbf(x1, x2)
        elif self.kernel == 'sigmoid':
            return Kernels.sigmoid(x1, x2)
        elif self.kernel == 'polynomial':
            return Kernels.polynomial(x1, x2)
        elif self.kernel == 'cosine':
            return Kernels.cosine(x1, x2)
        elif self.kernel == 'correlation':
            return Kernels.correlation(x1, x2)
        elif self.kernel == 'linrbf':
            return Kernels.linrbf(x1, x2)
        elif self.kernel == 'rbfpoly':
            return Kernels.rbfpoly(x1, x2)
        elif self.kernel == 'rbfcosine':
            return Kernels.rbfpoly(x1, x2)
        
    def fit(self, X):
        '''
        param: X: NxD
        '''
        self.X = X
        #normalized kernel
        self.normKernel = self.kernelize(X, X) - 2*1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(self.kernelize(X, X)) + \
                            1/X.shape[0]*np.ones((X.shape[0], X.shape[0])).dot(np.dot(1/X.shape[0]*np.ones((X.shape[0], X.shape[0])), self.kernelize(X, X)))
        self.eival, self.eivect = np.linalg.eig(self.normKernel)
        #sort eigen values and return explained variance
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        self.explained_variance = self.explained_variance_()
        #return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        return self
    
    
    def fit_transform(self):
        '''
        Return: transformed data
        '''
        return self.kernelize(self.X, self.X).dot(self.eivect)
    
#%% Testing

# X, y = load_hoda(train_size=1000, size=5)
# X, y = make_s_curve(n_samples=100, random_state=123)
# X, y = make_moons(n_samples=100, random_state=123)
X, y = load_iris().data, load_iris().target

kpca = kPCA(k=2, kernel='linrbf').fit(X)
kpca.explained_variance
newX = kpca.fit_transform()

"""
fig = plt.figure()
fig.suptitle('Classification with PCA - Curves dataset', ha='left', fontsize=11)
plt.subplot(2, 2, 1, aspect='equal')
plt.title('Original Space')
reds = y == 0
blues = y == 1
plt.scatter(X[reds, 0], X[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(newX[reds, 0], newX[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(newX[blues, 0], newX[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")

plt.savefig('/Users/samaneh/Desktop/images/curves-PCA.png', dpi=300)
plt.show()
"""


plt.scatter(newX[:, 0], newX[:, 1], c=y)
plt.title('Classification with PCA -  iris dataset')
plt.savefig('/Users/samaneh/Desktop/images/iris-KPCA.png', dpi=300)
plt.show()

