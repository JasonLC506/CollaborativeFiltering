"""
implement multiclass matrix factorization algorithms for CF data using Stochastic Gradient Descent according to
Menon, A.K. and Elkan, C., 2010, December. A log-linear model with latent features for dyadic prediction. In Data Mining (ICDM), 2010 IEEE 10th International Conference on (pp. 364-373). IEEE. [1]
"""

import numpy as np

class MF(object):
    def __init__(self):
        self.K = 0      # dim of embedding vectors
        self.N = 0      # num of users
        self.M = 0      # num of items
        self.L = 0      # num of classes in multiclass
        self.u = None   # embedding vectors for user
        self.v = None   # embedding vectors for items

    def fit(self, training, valid, K, max_epoch = 10):
        self.K = K
        self.N = training.N()   # from training get num of users
        self.M = training.M()   # from training get num of items
        self.L = training.L()   # from training get num of classes

        self.initialize()
        loss_valid = self.loss(valid)
        for epoch in range(max_epoch):
            for samp in training.sample():
                self.update(samp)
            loss_valid_new = self.loss(valid)
            if loss_valid_new > loss_valid:
                print "overfitting in epoch: ", epoch
                break

        return self

    def initialize(self):
        ## according to [1] ##
        self.u = np.random.normal(0.0, 1.0, size = self.L * self.N * self.K).reshape([self.L, self.N, self.K])
        self.v = np.random.normal(0.0, 1.0, size = self.L * self.M * self.K).reshape([self.L, self.M, self.K])

        return self

    def update(self, instance):
        """
        update embeddings according to a single instance with SGD
        """
        pass
        return self

    def loss(self, test):
        """
        calculate prediction performance in test data
        """
        pass
        return None



