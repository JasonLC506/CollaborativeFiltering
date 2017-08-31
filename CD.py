"""
Multiclass dyadic data learning
here Canonical Decomposition (CD) for tensor decomposition is employed
in context of facebook emoticon rating, rating classes are fixed, given and small
"""

import numpy as np
import cPickle
from MultiClassCF import MCCF


class CD(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.k = 0

        # paras predefined #
        self.r = None

        # paras ascome #
        self.u = {}
        self.v = {}

        # paras ascome default #
        self.u_avg = None
        self.v_avg = None

        # results storage #
        self.logfilename = "logs/CD"
        self.modelconfigurefile = "modelconfigures/CD_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = model_hyperparameters[0]

    def basicInitialize(self):
        self.r = np.random.normal(0.0, self.SCALE, size=(self.L, self.k))
        return self

    def initialize(self, uid, iid, predict=False):
        ## according to [1] in MultiMF ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, self.SCALE, size = self.k)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, self.SCALE, size = self.k)
        return self

    def update(self, instance):
        uid, iid, lid = instance
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        mgrad = expm / expmsum
        mgrad[lid] = mgrad[lid] - 1.0
        mgrad = - mgrad
        # gradient for embeddings #
        delt_u = np.tensordot(mgrad, np.multiply(self.r, self.v[iid]), axes = (0,0))
        delt_v = np.tensordot(mgrad, np.multiply(self.r, self.u[uid]), axes = (0,0))
        delt_r = np.outer(mgrad, np.multiply(self.u[uid], self.v[iid]))
        # update #
        self.u[uid] += (self.SGDstep * (delt_u))
        self.v[iid] += (self.SGDstep * (delt_v))
        self.r += (self.SGDstep * (delt_r))
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis = 0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis = 0)

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)

    def modelconfigStore(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {"u": self.u, "v": self.v, "r": self.r, "u_avg": self.u_avg, "v_avg": self.v_avg}
        with open(modelconfigurefile, "w") as f:
            cPickle.dump(paras, f)

    def modelconfigLoad(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        with open(modelconfigurefile, "r") as f:
            paras = cPickle.load(f)
        # write to model parameters #
        self.u = paras["u"]
        self.v = paras["v"]
        self.r = paras["r"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.r.shape[0]
        print "model successfully loaded from " + modelconfigurefile