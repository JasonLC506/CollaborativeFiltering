"""
pairwise interaction tensor factorization (PITF) model,
pairwise comapison among tags and pairwise decomposition of tensor by user-tag and item-tag additively, based on
Rendle, S. and Schmidt-Thieme, L., 2010, February. Pairwise interaction tensor factorization for personalized tag recommendation. In Proceedings of the third ACM international conference on Web search and data mining (pp. 81-90). ACM. [1]
"""

import numpy as np
import cPickle
from MultiClassCF import MCCF

class PITF(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.k = 0

        # paras predefined #
        self.r_u = None
        self.r_v = None

        # paras ascome #
        self.u = {}
        self.v = {}

        # paras ascome default #
        self.u_avg = None
        self.v_avg = None

        self.lamda = 0.0

        # results storage #
        self.logfilename = "logs/PITF"
        self.modelconfigurefile = "modelconfigures/PITF_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = model_hyperparameters[0]

    def basicInitialize(self):
        self.r_u = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))
        self.r_v = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))

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
        lid_neg_list = [i for i in range(self.L)]
        del lid_neg_list[lid]

        ### test ###
        loss_before = self.lossSingle(instance)

        m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
        delt_u = np.zeros(self.k, dtype=np.float64)
        delt_v = np.zeros(self.k, dtype=np.float64)
        delt_r_u = np.zeros([self.L, self.k], dtype=np.float64)
        delt_r_v = np.zeros([self.L, self.k], dtype=np.float64)

        for lid_neg in lid_neg_list:
            delt = (1.0 - sigmoid(m[lid] - m[lid_neg]))
            delt_u += (delt * (self.r_u[lid] - self.r_u[lid_neg]) - self.lamda * self.u[uid])
            delt_v += (delt * (self.r_v[lid] - self.r_v[lid_neg]) - self.lamda * self.v[iid])
            delt_r_u[lid] += (delt * self.u[uid] - self.lamda * self.r_u[lid])
            delt_r_u[lid_neg] += (- delt * self.u[uid] - self.lamda * self.r_u[lid_neg])
            delt_r_v[lid] += (delt * self.v[iid] - self.lamda * self.r_v[lid])
            delt_r_v[lid_neg] += (- delt * self.v[iid] - self.lamda * self.r_v[lid_neg])

        # update #
        self.u[uid] += (self.SGDstep * delt_u)
        self.v[iid] += (self.SGDstep * delt_v)
        self.r_u += (self.SGDstep * delt_r_u)
        self.r_v += (self.SGDstep * delt_r_v)

        ### test ###
        loss_after = self.lossSingle(instance)
        print "loss_before", loss_before, "loss_after", loss_after
        if loss_after > loss_before:
            print "single loss increase", loss_before, loss_after
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis=0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis=0)

    def loss(self, test):
        losssum = 0.0
        Nsamp = 0
        for samp in test.sample(random=False):
            losssum += self.lossSingle(samp)
            Nsamp += 1
        return losssum / Nsamp

    def lossSingle(self, instance):
        uid, iid, lid = instance
        self.initialize(uid, iid, predict=True)
        m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
        lid_neg_list = [i for i in range(self.L)]
        del lid_neg_list[lid]
        loss = 0.0
        for lid_neg in lid_neg_list:
            loss += np.log(sigmoid(m[lid] - m[lid_neg]))
        loss = - loss
        return loss

    def predict(self, uid, iid, distribution = False):
        if distribution:
            raise ValueError("PITF does not support distribution output")
        self.initialize(uid, iid, predict=True)
        m = np.tensordot(self.r_u, self.u[uid], axes = (1,0)) + np.tensordot(self.r_v, self.v[iid], axes = (1,0))
        return np.argmax(m)

    def modelconfigStore(self, modelconfigurefile = None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {"u": self.u, "v": self.v, "r_u": self.r_u, "r_v": self.r_v, "u_avg": self.u_avg, "v_avg": self.v_avg}
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
        self.r_u = paras["r_u"]
        self.r_v = paras["r_v"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.r_u.shape[0]
        print "model successfully loaded from " + modelconfigurefile

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))