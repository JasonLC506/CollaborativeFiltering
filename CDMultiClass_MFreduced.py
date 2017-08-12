"""
Multiclass dyadic data learning
here Canonical Decomposition (CD) for tensor decomposition is employed
in context of facebook emoticon rating, rating classes are fixed, given and small
"""

import numpy as np
from TDMultiClass import TD

SCALE = 1.0

class CD(TD):
    def __init__(self):
        TD.__init__(self)
        # model hyperparameters #
        self.k = 0

        # # fitting intermediate #
        # self.delt_r_batch = None
        #
        # # fitting hyperparameter #
        # self.mini_batch = 500

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
                loss_single_before = self.lossSingle(samp)  ###
                self.update(samp, cnt)
                cnt += 1
                loss_single_after = self.lossSingle(samp)   ###
                if loss_single_after > loss_single_before:
                    print loss_single_before, "to", loss_single_after
            self.averageEmbedding()
            loss_train = self.loss(training)
            loss_valid_new = self.loss(valid)
            print "after epoch ", epoch, "loss training: ", loss_train
            print "after epoch ", epoch, "loss valid: ", loss_valid_new
            if loss_valid is not None and loss_valid_new > loss_valid:
                print "overfitting in epoch: ", epoch
                # break
            loss_valid = loss_valid_new
            # self.SGDstepUpdate(epoch)
        return self

    def basicInitialize(self):
        self.r = np.random.normal(0.0, SCALE, size = (self.L, self.k))
        ### test ###
        # self.delt_r_batch = np.zeros(self.r.shape, dtype = np.float64)
        # self.r = np.ones([self.L, self.k], dtype=np.float64)
        step = self.k / self.L
        for i in range(self.L):
            self.r[i, (i+1)*step:self.k] = 0.0
            self.r[i, 0:i*step] = 0.0
        # print "self.r", self.r
        return self

    def initialize(self, uid, iid, predict=False):
        ## according to [1] in MFMultiClass ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, SCALE, size = self.k)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, SCALE, size = self.k)
        return self

    def update(self, instance, isamp):
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
        delt_u = np.tensordot(mgrad, np.multiply(self.r, self.v[iid]), axes = (0,0))
        delt_v = np.tensordot(mgrad, np.multiply(self.r, self.u[uid]), axes = (0,0))
        delt_r = np.outer(mgrad, np.multiply(self.u[uid], self.v[iid]))
        ### test ###
        rela_u = np.linalg.norm(delt_u)/np.linalg.norm(self.u[uid])
        rela_v = np.linalg.norm(delt_v)/np.linalg.norm(self.v[iid])
        rela_r = np.linalg.norm(delt_r)/np.linalg.norm(self.r)
        if rela_u > 0.1 / self.SGDstep or rela_v > 0.1 / self.SGDstep or rela_r > 0.1 / self.SGDstep:
            print "large step"
            print rela_u
            print rela_v
            print rela_r
        # update #
        self.u[uid] += (self.SGDstep * (delt_u))
        self.v[iid] += (self.SGDstep * (delt_v))
        # self.r += (self.SGDstep * (delt_r))
        # ### test ###
        # step = self.k / self.L
        # for i in range(self.L):
        #     self.r[i, (i+1)*step:self.k] = 0.0
        #     self.r[i, 0:i*step] = 0.0

        return self

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)
