#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 08:05:26 2019

@author: Samaneh
"""


from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.datasets import make_s_curve
from AML.dataset import load_hoda


class PCA:
    def __init__(self, k=None):
        """
        :param: k: Number of principal components to select
                    Default value is 2.
        """
        if not k:
            k = 2
            self.k = k
        else:
            self.k = k
        return
    
    def explained_variance_(self):
        """
        :Return: explained variance. (eigenvalues importance calculated)
        """
        self.total_eigenvalue = np.sum(self.eival)
        self.explained_variance = [x/self.total_eigenvalue*100 for x in sorted(self.eival, reverse=True)[:self.k]]
        return self.explained_variance
    
    def fit(self, X):
        """
        :param: X: NxD (Covariance Matrix, eigen vectors and eigen values are calculated.)
        """
        self.X = X
        # centered mean
        self.X = self.X - np.mean(self.X, axis=0)
        # covariance
        self.cov = (1/self.X.shape[1])* np.dot(self.X.T, self.X)
        self.eival, self.eivect = np.linalg.eig(self.cov)
        self.sorted_eigen = np.argsort(self.eival[:self.k])[::-1]
        # sort eigen values and return explained variance
        self.explained_variance = self.explained_variance_()
        # return eigen value and corresponding eigenvectors
        self.eival, self.eivect = self.eival[:self.k], self.eivect[:, self.sorted_eigen]
        return self
    
    def fit_transform(self):
        """
        :Return: transformed datapoints
        """
        return self.X.dot(self.eivect)

#%% Testing

# X, y = load_hoda(train_size=1000, size=5)
# X, y = make_s_curve(n_samples=100, random_state=123)
# X, y = make_moons(n_samples=100, random_state=123)
X, y = load_iris().data, load_iris().target
# A = np.array([[1, 2], [3, 4], [5, 6]])
pca = PCA(k=2).fit(X)
print(pca.explained_variance)
newX = pca.fit_transform()

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
plt.savefig('/Users/samaneh/Desktop/images/iris-PCA.png', dpi=300)
plt.show()








