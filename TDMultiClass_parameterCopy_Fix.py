"""
Multiclass dyadic data learning
here Tucker Decomposition (TD) for tensor decomposition is employed
in context of facebook emoticon rating, rating classes are fixed, given and small
"""

import numpy as np
import matrixTool
from MFMultiClass import ppl
import cPickle

SCALE = 0.1

class TD(object):
    def __init__(self):
        # model hyperparameters #
        self.ku = 0     # dim for embedding of users
        self.kv = 0     # dim for embedding of items
        self.kr = 0     # dim for embedding of rating classes

        # model parameters #
        self.L = 0      # size of rating class set
        self.c = None   # core matrix of TD, np.array([self.ku, self.kv, self.kr])
        self.u = {}     # embedding vectors of users
        self.v = {}     # embedding vectors of items
        self.r = None     # embedding vectors of rating classes, np.array([self.L, self.kr])

        self.u_avg = None   # avg embeddings of users, default for cold-start
        self.v_avg = None   # avg embeddings of users, default for cold-start

        # fitting hyperparameters #
        self.SGDstep = 0.001
        self.SGDstep_step = 0.8
        self.lamda = 0.001

    def fit(self, training, valid, ku, kv, kr, max_epoch = 100, SGDstep = 0.001):
        # set parameters #
        self.ku = ku
        self.kv = kv
        self.kr = kr
        self.L = training.L()
        self.SGDstep = SGDstep

        self.basicInitialize()

        # SGD #
        loss_valid = None
        for epoch in xrange(max_epoch):
            for samp in training.sample():
                self.update(samp)
            loss_train = self.loss(training)
            loss_valid_new = self.loss(valid)
            print "after epoch ", epoch, "loss training: ", loss_train
            print "after epoch ", epoch, "loss valid: ", loss_valid_new
            if loss_valid is not None and loss_valid_new > loss_valid:
                print "overfitting in epoch: ", epoch
                break
            loss_valid = loss_valid_new
            # self.SGDstepUpdate(epoch)

        return self

    def basicInitialize(self):
        with open("data")
        return self


    def update(self, instance):
        """
        instance: [UserId, ItemId, LabelId] where LabelID should be index in range(self.L)
        """
        uid, iid, lid = instance
        ## calculate single gradient ##
        # intermediate #
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        expm = np.exp(m)
        expmsum = np.sum(expm)
        mgrad = expm / expmsum
        mgrad[lid] = mgrad[lid] - 1.0
        mgrad = - mgrad
        # gradient for embeddings #
        delt_u = np.tensordot(a = mgrad, axes = (0,1),
                              b = np.tensordot(a = self.v[iid], axes = (0,1),
                                               b = np.tensordot(a = self.c, axes = (2,1),
                                                                b = self.r)
                                               )
                              )
        delt_v = np.tensordot(a = mgrad, axes = (0,1),
                              b = np.tensordot(a = self.u[uid], axes = (0,0),
                                               b = np.tensordot(a = self.c, axes = (2,1),
                                                                b = self.r)
                                               )
                              )
        delt_c = matrixTool.TensorOuter([self.u[uid], self.v[iid], np.tensordot(mgrad, self.r, axes = (0,0))])
        delt_r = np.outer(mgrad, np.tensordot(a = self.u[uid], axes = (0,0),
                                              b = np.tensordot(a = self.v[iid], axes = (0,1),
                                                               b = self.c)))
        ### test ###
        # rela_u = np.linalg.norm(delt_u)/np.linalg.norm(self.u[uid])
        # rela_v = np.linalg.norm(delt_v)/np.linalg.norm(self.v[iid])
        # rela_c = np.linalg.norm(delt_c)/np.linalg.norm(self.c)
        # rela_r = np.linalg.norm(delt_r)/np.linalg.norm(self.r)
        # print "relative step size", rela_u, rela_v, rela_c, rela_r
        # update #
        self.u[uid] += (self.SGDstep * (delt_u - self.lamda * self.u[uid]))
        self.v[iid] += (self.SGDstep * (delt_v - self.lamda * self.v[iid]))
        self.c += (self.SGDstep * (delt_c - self.lamda * self.c))
        self.r += (self.SGDstep * (delt_r - self.lamda * self.r))
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

    def lossSingle(self, instance):
        uid, iid, lid = instance
        predprob = self.predict(uid, iid, distribution = True)
        perf = ppl(predprob = predprob, truelabel = lid)
        return perf

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)

    def SGDstepUpdate(self, epoch):
        self.SGDstep = self.SGDstep * self.SGDstep_step
        print "SGDstep for next epoch %d: %f" % (epoch, self.SGDstep)

def TDreconstruct(c, u, v, t):
    ## calculate TD reconstruction with t as matrix np.array(self.L, self.kr) ##
    m = np.tensordot(a = t, axes = (1, 0),
                     b = np.tensordot(a = u, axes = (0, 0),
                                    b = np.tensordot(a = v, axes = (0, 1),
                                                   b = c)
                                    )
                     )  # np.array([self.L,])
    return m