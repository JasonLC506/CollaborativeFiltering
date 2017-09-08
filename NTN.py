"""
adopt knowledge graph relation classification algo, Neural Tensor Network
Socher, R., Chen, D., Manning, C. D., & Ng, A. (2013). Reasoning with neural tensor networks for knowledge base completion. In Advances in neural information processing systems (pp. 926-934).[1]
"""

import numpy as np
import cPickle
from matrixTool import transMultiply
from matrixTool import TensorOuterFull
from MultiClassCF import MCCF

class NTN(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.k = 0
        self.d = 0

        # paras_predefined #
        self.W2 = None
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
        self.logfilename = "logs/NTN"
        self.modelconfigurefile = "modelconfigures/NTN_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 2:
            raise ValueError("number of hyperparameters wrong")
        self.k, self.d = model_hyperparameters

    def basicInitialize(self):
        self.W2 = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))
        self.W1bi = np.random.normal(0.0, self.SCALE, size = (self.L, self.k, self.d, self.d))
        self.W1u = np.random.normal(0.0, self.SCALE, size = (self.L, self.k, self.d))
        self.W1v = np.random.normal(0.0, self.SCALE, size = (self.L, self.k, self.d))
        self.B1 = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))
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
        outL1 = self._outL1(L1)
        m = np.sum(np.multiply(self.W2, outL1), axis = 1)
        mgrad = self._softmaxGradient(m, lid)
        L1grad = transMultiply(np.multiply(self._gradL1(L1), self.W2), mgrad)
        # gradient #
        delt_W2 = transMultiply(outL1, mgrad)
        delt_W1bi = TensorOuterFull([L1grad, self.u[uid], self.v[iid]])
        delt_W1u = TensorOuterFull([L1grad, self.u[uid]])
        delt_W1v = TensorOuterFull([L1grad, self.v[iid]])
        delt_B1 = 1.0 * L1grad
        delt_u = np.tensordot(a = L1grad.reshape(self.L * self.k), axes = (0,0),
                              b = (np.tensordot(self.W1bi, self.v[iid], axes=(-1,0)) + self.W1u).reshape([(self.L * self.k), self.d]))
        delt_v = np.tensordot(a = L1grad.reshape(self.L * self.k), axes = (0,0),
                              b = (np.tensordot(self.W1bi, self.u[uid], axes=(-2,0)) + self.W1v).reshape([(self.L * self.k), self.d]))
        # update #
        self.W2 += (self.SGDstep * (delt_W2 - self.lamda * self.W2))
        self.W1bi += (self.SGDstep * (delt_W1bi - self.lamda * self.W1bi))
        self.W1u += (self.SGDstep * (delt_W1u - self.lamda * self.W1u))
        self.W1v += (self.SGDstep * (delt_W1v - self.lamda * self.W1v))
        self.B1 += (self.SGDstep * (delt_B1 - self.lamda * self.B1))
        self.u[uid] += (self.SGDstep * (delt_u - self.lamda * self.u[uid]))
        self.v[iid] += (self.SGDstep * (delt_v - self.lamda * self.v[iid]))
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis=0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis=0)

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        L1 = self._L1(uid, iid)
        outL1 = self._outL1(L1)
        m = np.sum(np.multiply(self.W2, outL1), axis = 1)
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)

    def modelconfigStore(self, modelconfigurefile = None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {
                 "W2": self.W2,
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
        self.W2 = paras["W2"]
        self.W1bi = paras["W1bi"]
        self.W1u = paras["W1u"]
        self.W1v = paras["W1v"]
        self.B1 = paras["B1"]
        self.u = paras["u"]
        self.v = paras["v"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.W2.shape[0]
        print "model successfully loaded from " + modelconfigurefile

    def _L1(self, uid, iid):
        bi = np.tensordot(a = self.u[uid], axes = (0,-1),
                          b = np.tensordot(a = self.v[iid], axes = (0,-1),
                                           b = self.W1bi))
        lu = np.tensordot(a = self.W1u, b = self.u[uid], axes = (-1,0))
        lv = np.tensordot(a = self.W1v, b = self.v[iid], axes = (-1,0))
        return (bi + lu + lv + self.B1)

    def _outL1(self, L1):
        return np.tanh(L1)

    def _gradL1(self, L1):
        return np.power(np.cosh(L1), -2)

    def _softmaxGradient(self, m, lid):
        expm = np.exp(m)
        expmsum = np.sum(expm)
        mgrad = expm / expmsum
        mgrad[lid] = mgrad[lid] - 1.0
        mgrad = - mgrad
        return mgrad