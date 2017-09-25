"""
Multiclass dyadic data learning
here Tucker Decomposition (TD) for tensor decomposition is employed
in context of facebook emoticon rating, rating classes are fixed, given and small
"""

import numpy as np
import cPickle
import matrixTool
from MultiClassCF import MCCF, softmaxOutput, softmaxGradient


class TD(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.ku = 0
        self.kv = 0
        self.kr = 0

        # paras_predefined #
        self.c = None
        self.r = None

        # paras_ascome #
        self.u = {}
        self.v = {}

        # paras_ascome_default #
        self.u_avg = None
        self.v_avg = None

        # results storage #
        self.logfilename = "logs/TD"
        self.modelconfigurefile = "modelconfigures/TD_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 3:
            raise ValueError("number of hyperparameters wrong")
        self.ku, self.kv, self.kr = model_hyperparameters

    def basicInitialize(self):
        self.r = np.random.normal(0.0, self.SCALE, size = (self.L, self.kr))
        self.c = np.random.normal(0.0, self.SCALE, size = (self.ku, self.kv, self.kr))
        return self

    def initialize(self, uid, iid, predict=False):
        ## according to [1] in MFMultiClass ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, self.SCALE, size = self.ku)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, self.SCALE, size = self.kv)
        return self

    def update(self, instance):
        """
        instance: [UserId, ItemId, LabelId] where LabelID should be index in range(self.L)
        """
        uid, iid, lid = instance
        ## calculate single gradient ##
        # intermediate #
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        mgrad = softmaxGradient(m, lid)
        # gradient for embeddings #
        delt_u = np.tensordot(a=mgrad, axes=(0, 1),
                              b=np.tensordot(a=self.v[iid], axes=(0, 1),
                                             b=np.tensordot(a=self.c, axes=(2, 1),
                                                            b=self.r)
                                             )
                              )
        delt_v = np.tensordot(a=mgrad, axes=(0, 1),
                              b=np.tensordot(a=self.u[uid], axes=(0, 0),
                                             b=np.tensordot(a=self.c, axes=(2, 1),
                                                            b=self.r)
                                             )
                              )
        delt_c = matrixTool.TensorOuter([self.u[uid], self.v[iid], np.tensordot(mgrad, self.r, axes=(0, 0))])
        delt_r = np.outer(mgrad, np.tensordot(a=self.u[uid], axes=(0, 0),
                                              b=np.tensordot(a=self.v[iid], axes=(0, 1),
                                                             b=self.c)))
        # update #
        self.u[uid] += (self.SGDstep * (delt_u - self.lamda * self.u[uid]))
        self.v[iid] += (self.SGDstep * (delt_v - self.lamda * self.v[iid]))
        self.c += (self.SGDstep * (delt_c - self.lamda * self.c))
        self.r += (self.SGDstep * (delt_r - self.lamda * self.r))

        ### test ###
        # m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        # print "m after update", m
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis = 0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis = 0)

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        return softmaxOutput(m, distribution=distribution)

    def modelconfigStore(self, modelconfigurefile = None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {"c": self.c, "u": self.u, "v": self.v, "r": self.r, "u_avg": self.u_avg, "v_avg": self.v_avg}
        with open(modelconfigurefile, "w") as f:
            cPickle.dump(paras, f)

    def modelconfigLoad(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        with open(modelconfigurefile, "r") as f:
            paras = cPickle.load(f)
        # write to model parameters #
        self.c = paras["c"]
        self.u = paras["u"]
        self.v = paras["v"]
        self.r = paras["r"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.r.shape[0]
        print "model successfully loaded from " + modelconfigurefile


def TDreconstruct(c, u, v, t):
    ## calculate TD reconstruction with t as matrix np.array(self.L, self.kr) ##
    m = np.tensordot(a = t, axes = (1, 0),
                     b = np.tensordot(a = u, axes = (0, 0),
                                    b = np.tensordot(a = v, axes = (0, 1),
                                                   b = c)
                                    )
                     )  # np.array([self.L,])
    return m