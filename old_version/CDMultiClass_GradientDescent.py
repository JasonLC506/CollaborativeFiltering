"""
Multiclass dyadic data learning
here Canonical Decomposition (CD) for tensor decomposition is employed
in context of facebook emoticon rating, rating classes are fixed, given and small
"""

import numpy as np
from TDMultiClass import TD

class CD(TD):
    def __init__(self):
        TD.__init__(self)
        # model hyperparameters #
        self.k = 0

        self.delt_u = {}
        self.delt_v = {}
        self.delt_r = None

    def fit(self, training, valid, K, max_epoch = 100, SGDstep = 0.001):
        self.k = K
        self.L = training.L()
        self.SGDstep = SGDstep

        self.basicInitialize()

        # SGD #
        loss_valid = None
        for epoch in xrange(max_epoch):
            cnt = 0
            for samp in training.sample():
                uid, iid, lid = samp
                self.initialize(uid, iid)
                # loss_single_before = self.lossSingle(samp)  ###
                self.stepupdate(samp, cnt)
                cnt += 1
                # loss_single_after = self.lossSingle(samp)   ###
                # if loss_single_after > loss_single_before:
                #     print loss_single_before, "to", loss_single_after
            self.averageEmbedding()
            loss_train_old = self.loss(training)
            loss_valid_old = self.loss(valid)
            self.update(cnt)
            self.averageEmbedding()
            loss_train = self.loss(training)
            loss_valid = self.loss(valid)
            print "before epoch", epoch, "loss training: ", loss_train_old
            print "after epoch ", epoch, "loss training: ", loss_train
            print "before epoch", epoch, "loss valid: ", loss_valid_old
            print "after epoch ", epoch, "loss valid: ", loss_valid
            if loss_valid_old is not None and loss_valid > loss_valid_old:
                print "overfitting in epoch: ", epoch
                break
            if loss_train > loss_train_old:
                print "loss increase in training set, sucks!!"
                break
            # self.SGDstepUpdate(epoch)
        return self

    def basicInitialize(self):
        self.r = np.random.normal(0.0, 1.0, size = (self.L, self.k))
        self.delt_r = np.zeros([self.L, self.k], dtype=np.float64)
        return self

    def initialize(self, uid, iid, predict=False):
        ## according to [1] in MFMultiClass ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, 1.0, size = self.k)
            self.delt_u[uid] = np.zeros(self.k, dtype=np.float64)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, 1.0, size = self.k)
            self.delt_v[iid] = np.zeros(self.k, dtype=np.float64)
        return self

    def stepupdate(self, instance, isamp):
        uid, iid, lid = instance
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        mgrad = expm / expmsum
        mgrad[lid] = mgrad[lid] - 1.0
        mgrad = - mgrad
        try:
            assert abs(np.sum(mgrad)) < 1e-5
        except:
            print "mgrad error"
            print mgrad
        # gradient for embeddings #
        self.delt_u[uid] += np.tensordot(mgrad, np.multiply(self.r, self.v[iid]), axes = (0,0))
        self.delt_v[iid] += np.tensordot(mgrad, np.multiply(self.r, self.u[uid]), axes = (0,0))
        self.delt_r += np.outer(mgrad, np.multiply(self.u[uid], self.v[iid]))
        # # update #
        # self.u[uid] += (self.SGDstep * (delt_u))
        # self.v[iid] += (self.SGDstep * (delt_v))
        # ### test ###
        # self.r += (self.SGDstep * (delt_r))

        return self

    def update(self, Nsamp):
        for uid in self.delt_u.keys():
            self.u[uid] += (self.SGDstep * self.delt_u[uid] / Nsamp)
            del self.delt_u[uid]
            self.delt_u[uid] = np.zeros(self.k, dtype=np.float64)
        for iid in self.delt_v.keys():
            self.v[iid] += (self.SGDstep * self.delt_v[iid] / Nsamp)
            del self.delt_v[iid]
            self.delt_v[iid] = np.zeros(self.k, dtype=np.float64)
        self.r += (self.SGDstep * self.delt_r / Nsamp)
        del self.delt_r
        self.delt_r = np.zeros([self.L, self.k], dtype=np.float64)


    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)
