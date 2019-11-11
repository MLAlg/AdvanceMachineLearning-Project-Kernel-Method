#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 01:41:45 2019

@author: Samaneh
"""

from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import copy
from AML.Utils.kernels import Kernels
from sklearn.datasets import make_blobs, make_moons, make_circles
import cv2
from AML.HodaDatasetReader import read_hoda_cdb


class kkmeans(Kernels):
    def __init__(self, k=None, kernel=None):
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
    
    def kernelize(self, x1, x2):
        """
        :params: x1: NxD
        :params: x2: NxD
        """
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
        
    def distance(self, x, nu):
        """
        :param: x: datapoint
        :param: nu: mean
        :retrun distance matrix
        """
       
        return self.kernelize(x, x) + self.kernelize(nu, nu) - 2*self.kernel(x, nu)
    
    
    def fit(self, X):
        """
        :param: X: NxD
        """
        self.X = X
        # random sample
        N, D = X.shape
        # randomly initialize k centroids
        self.nu = X[np.random.choice(N, self.k, replace = False)]
        self.prev_c = np.zeros((self.k, D))
        self.cluster = np.zeros(X.shape[0])
        '''iterate by checking to see if new centroid
        of new center is same as old center, then we reached an
        optimum point.
        '''
        while self.distance(self.X, self.nu) != 0:
            for ii in range(X.shape[0]):
                self.distance_matrix = self.distance(self.X[ii], self.nu)
                self.cluster[ii] = np.argmin(self.distance_matrix)
            self.prev_c = copy.deepcopy(self.nu)
            for ij in range(self.k):
                # mean of the new found clusters
                self.newPoints = [X[ii] for ii in range(X.shape[0]) if self.cluster[ii] == ij]
                self.nu[ij] = np.mean(self.newPoints, axis = 0)
        return self
    
    def predict(self, X):
        """
        :param: X: NxD
        :return type: labels
        """
        pred = np.zeros(X.shape[0])
        # compare new data to final centroid
        for ii in range(X.shape[0]):
            distance_matrix = self.distance(X[ii], self.nu)
            pred[ii] = np.argmin(distance_matrix)
        return pred
    

#%% Testing


print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('AML/DigitDB/Train 60000.cdb')
print("train shape : ", np.shape(train_images[1]))
dataset = []
for idx, image in enumerate(train_images):
    dataset.append(cv2.resize(image, (50, 50)))

dataset = np.reshape(dataset, (60000, 2500))
dataset = dataset[1:60000]
np.shape(dataset)
dataset = dataset.astype('float32') / 255.

X = dataset[1:1000]
newX = dataset[1000:1200]

#X, y = make_blobs(n_samples=100, random_state=123)
# X, y = make_moons(n_samples=100, random_state=123)
# X, y = make_circles(1000, noise=.07, factor=.5)
# X = np.array([[1, 2], [1, 4], [1, 0],
#              [10, 2], [10, 4], [10, 0]])
# new_x = np.array([[1, 5], [10, 6], [10, 3]])
kernelkmns = kkmeans().fit(X)
pred = kernelkmns.predict(newX)

#plt.scatter(X[:, 0], X[:, 1], c=kernelkmns.cluster)
#plt.scatter(kernelkmns.nu[:, 0], kernelkmns.nu[:, 1], marker='.')
#plt.show()

