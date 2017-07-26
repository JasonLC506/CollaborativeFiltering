"""
implement multiclass matrix factorization algorithms for CF data using Stochastic Gradient Descent according to
Menon, A.K. and Elkan, C., 2010, December. A log-linear model with latent features for dyadic prediction. In Data Mining (ICDM), 2010 IEEE 10th International Conference on (pp. 364-373). IEEE. [1]
"""

import numpy as np

class MF(object):
    def __init__(self):
        self.K = 0      # dim of embedding vectors
        # self.N = 0      # num of users
        # self.M = 0      # num of items
        self.L = 0      # num of classes in multiclass
        self.u = {}   # embedding vectors for user
        self.v = {}   # embedding vectors for items
        self.u_avg = None   # avg of embeddings of users, default for cold-start
        self.v_avg = None   # avg of embeddings of items, default for cold-start

        # hyperparameters #
        self.SGDstep = 0.001

    def fit(self, training, valid, K, max_epoch = 10, SGDstep = 0.001):
        self.K = K
        self.SGDstep = SGDstep
        # self.N = training.N()   # from training get num of users
        # self.M = training.M()   # from training get num of items
        self.L = training.L()   # from training get num of classes

        loss_valid = None

        for epoch in range(max_epoch):
            for samp in training.sample():
                uid, iid, lid = samp
                self.initialize(uid, iid) # initialize emdeddings if not occured before
                self.update(samp)
            self.averageEmbedding()
            loss_train = self.loss(training)
            loss_valid_new = self.loss(valid)
            print "after epoch ", epoch, "loss training: ", loss_train
            print "after epoch ", epoch, "loss valid: ", loss_valid_new
            if loss_valid is not None and loss_valid_new > loss_valid:
                print "overfitting in epoch: ", epoch
                break
            else:
                loss_valid = loss_valid_new

        return self

    def initialize(self, uid, iid, predict = False):
        ## according to [1] ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, 1.0, size = self.L * self.K).reshape([self.L, self.K])
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, 1.0, size = self.L * self.K).reshape([self.L, self.K])

        return self

    def update(self, instance):
        """
        update embeddings according to a single instance with SGD
        instance: [UserId, ItemId, LabelId]
        """
        uid, iid, lid = instance
        ## calculate update step ##
        # intermediate #
        expprod = np.exp(np.sum(np.multiply(self.u[uid], self.v[iid]), axis = 1))
        expprodsum = np.sum(expprod)
        # for u #
        delt_u = - np.transpose(np.multiply(np.transpose(self.v[iid]), expprod)) / expprodsum
        delt_u[lid] += self.v[iid][lid]
        # for v #
        delt_v = - np.transpose(np.multiply(np.transpose(self.u[uid]), expprod)) / expprodsum
        delt_v[lid] += self.u[uid][lid]

        self.u[uid] += (self.SGDstep * delt_u)
        self.v[iid] += (self.SGDstep * delt_v)
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis = 0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis = 0)

    def loss(self, test):
        """
        calculate prediction performance in test data
        """
        losssum = 0.0
        nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid = samp
            predprob = self.predict(uid, iid, distribution = True)
            perf = ppl(predprob = predprob, truelabel = lid)
            losssum += perf
            nsamp += 1
        return losssum/nsamp

    def predict(self, uid, iid, distribution = True):
        """
        predict rating matrix entry given userID and itemID,
        distribution == True when probability distribution is output
        """
        self.initialize(uid, iid, predict = True)   # set avg embeddings for cold-start entries
        expprod = np.exp(np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1))
        expprodsum = np.sum(expprod)
        if distribution:
            return expprod/expprodsum
        else:
            return np.argmax(expprod)


def ppl(predprob, truelabel):
    return (- np.log(predprob[truelabel]))



