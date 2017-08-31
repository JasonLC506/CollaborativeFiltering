"""
0,1 square loss version of TD
"""

import numpy as np
import matrixTool
from TD import TD
from TD import TDreconstruct

class TD01Loss(TD):
    def __init__(self):
        TD.__init__(self)

        # results storage #
        self.logfilename += "01Loss"
        self.modelconfigurefile = "modelconfigures/TD01Loss_config"

    def update(self, instance):
        uid, iid, lid = instance
        ## calculate single gradient ##
        # intermediate #
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        mgrad = -m
        mgrad[lid] = 1.0 + mgrad[lid]
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
        return self

    def loss(self, test):
        losssum = 0.0
        nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid = samp
            self.initialize(uid, iid, predict=True)
            m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
            m_true = np.zeros(self.L)
            m_true[lid] = 1.0
            losssum += np.sum(np.power((m - m_true), 2.0))
            nsamp += 1
        return losssum/nsamp