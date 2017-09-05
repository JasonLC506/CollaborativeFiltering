"""
Inspired by NTN, a Neural network model with bilinear layer,
simplified version as single layer
"""

import numpy as np
import cPickle
from matrixTool import transMultiply
from matrixTool import TensorOuterFull
from MultiClassCF import MCCF

class BiNNsingle(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.d = 0

        # paras_predefined #
        self.W1bi = None
        self.W1u = None
        self.W1v = None
        self.B1 = None

        # paras_ascome #
        self.u = {}
        self.v = {}

        # paras_ascome default #
        self.u_avg = None
        self.v_avg = None

        # results storage #
        self.logfilename = "logs/BiNNsingle"
        self.modelconfigurefile = "modelconfigures/BiNNsingle_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.d = model_hyperparameters[0]

    def basicInitialize(self):
        self.W1bi = np.random.normal(0.0, self.SCALE, size = (self.L, self.d, self.d))
        self.W1u = np.random.normal(0.0, self.SCALE, size = (self.L, self.d))
        self.W1v = np.random.normal(0.0, self.SCALE, size = (self.L, self.d))
        self.B1 = np.random.normal(0.0, self.SCALE, size = self.L)
        # ### test ###
        # # reduce to MultiMA #
        # assert self.W1u.shape[0] == self.W1u.shape[1]
        # self.W1bi[:,:,:] = 0.0
        # for i in range(self.L):
        #     self.W1u[i,:] = 0.0
        #     self.W1u[i,i] = 1.0
        #     self.W1v[i,:] = 0.0
        #     self.W1v[i,i] = 1.0
        # self.B1[:] = 0.0
        return self

    def initialize(self, uid, iid, predict = False):
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, self.SCALE, size = self.d)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, self.SCALE, size = self.d)
        return self

    def update(self, instance):
        uid, iid, lid = instance
        ## calculate single gradient ##
        # intermediate #
        L1 = self._L1(uid, iid)
        L1grad = softmaxGradient(L1, lid)
        # gradient #
        delt_W1bi = TensorOuterFull([L1grad, self.u[uid], self.v[iid]])
        delt_W1u = TensorOuterFull([L1grad, self.u[uid]])
        delt_W1v = TensorOuterFull([L1grad, self.v[iid]])
        delt_B1 = L1grad * 1.0
        delt_u = np.tensordot(a = L1grad, axes = (0,0),
                              b = (self.W1u + np.tensordot(self.W1bi, self.v[iid], axes=(-1,0))))
        delt_v = np.tensordot(a = L1grad, axes = (0,0),
                              b = (self.W1v + np.tensordot(self.W1bi, self.u[uid], axes=(-2,0))))
        # update #
        self.W1bi += (self.SGDstep * delt_W1bi)
        self.W1u += (self.SGDstep * delt_W1u)
        self.W1v += (self.SGDstep * delt_W1v)
        self.B1 += (self.SGDstep * delt_B1)
        self.u[uid] += (self.SGDstep * delt_u)
        self.v[iid] += (self.SGDstep * delt_v)
        # ### test ###
        # # reduce to MultiMA #
        # assert self.W1u.shape[0] == self.W1u.shape[1]
        # self.W1bi[:,:,:] = 0.0
        # for i in range(self.L):
        #     self.W1u[i,:] = 0.0
        #     self.W1u[i,i] = 1.0
        #     self.W1v[i,:] = 0.0
        #     self.W1v[i,i] = 1.0
        # self.B1[:] = 0.0
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis=0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis=0)

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        L1 = self._L1(uid, iid)
        return softmaxOutput(L1, distribution = distribution)

    def modelconfigStore(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {
            "W1bi": self.W1bi,
            "W1u": self.W1u,
            "W1v": self.W1v,
            "B1": self.B1,
            "u": self.u,
            "v": self.v,
            "u_avg": self.u_avg,
            "v_avg": self.v_avg
        }
        with open(modelconfigurefile, "w") as f:
            cPickle.dump(paras, f)

    def modelconfigLoad(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        with open(modelconfigurefile, "r") as f:
            paras = cPickle.load(f)
        # write to model parameters #
        self.W1bi = paras["W1bi"]
        self.W1u = paras["W1u"]
        self.W1v = paras["W1v"]
        self.B1 = paras["B1"]
        self.u = paras["u"]
        self.v = paras["v"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.W1u.shape[0]
        print "model successfully loaded from " + modelconfigurefile

    def _L1(self, uid, iid):
        bi = np.tensordot(a = self.u[uid], axes = (0,-1),
                          b = np.tensordot(a = self.v[iid], axes = (0,-1),
                                           b = self.W1bi))
        lu = np.tensordot(a = self.W1u, b = self.u[uid], axes = (-1,0))
        lv = np.tensordot(a = self.W1v, b = self.v[iid], axes = (-1,0))
        return (bi + lu + lv + self.B1)


def denseLayer(outLayerLower, W, B):
    return (np.tensordot(W, outLayerLower, axes=(-1,0)) + B)


def denseLayerGradBP(GradLayerUp, W):
    return np.tensordot(GradLayerUp, W, axes = (0,0))


def softmaxGradient(m, lid):
    expm = np.exp(m)
    expmsum = np.sum(expm)
    mgrad = expm / expmsum
    mgrad[lid] = mgrad[lid] - 1.0
    mgrad = - mgrad
    return mgrad


def softmaxOutput(m, distribution = True):
    expm = np.exp(m)
    expmsum = np.sum(expm)
    if distribution:
        return expm / expmsum
    else:
        return np.argmax(expm)