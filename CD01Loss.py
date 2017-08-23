"""
0,1 square loss version of CD
"""

import numpy as np
from CD import CD

class CD01Loss(CD):
    def __init__(self):
        CD.__init__(self)

        # results storage #
        self.logfilename += "01Loss"
        self.modelconfigurefile = "modelconfigures/CD01Loss_config"

    def update(self, instance):
        uid, iid, lid = instance
        m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes=(1, 0))
        mgrad = - m
        mgrad[lid] = 1.0 + mgrad[lid]
        # gradient for embeddings #
        delt_u = np.tensordot(mgrad, np.multiply(self.r, self.v[iid]), axes=(0, 0))
        delt_v = np.tensordot(mgrad, np.multiply(self.r, self.u[uid]), axes=(0, 0))
        delt_r = np.outer(mgrad, np.multiply(self.u[uid], self.v[iid]))
        # update #
        self.u[uid] += (self.SGDstep * (delt_u))
        self.v[iid] += (self.SGDstep * (delt_v))
        self.r += (self.SGDstep * (delt_r))
        return self

    def loss(self, test):
        losssum = 0.0
        Nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid = samp
            self.initialize(uid, iid, predict = True)
            m = np.tensordot(self.r, np.multiply(self.v[iid], self.u[uid]), axes = (1,0))
            m_true = np.zeros(self.L)
            m_true[lid] = 1.0
            losssum += np.sum(np.power((m - m_true),2.0))
            Nsamp += 1
        return losssum / Nsamp