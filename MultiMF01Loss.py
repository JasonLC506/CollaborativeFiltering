"""
0,1 square loss version of MultiMF
"""

import numpy as np
from MultiMF import MultiMF

class MultiMF01Loss(MultiMF):
    def __init__(self):
        MultiMF.__init__(self)

        # results storage #
        self.logfilename += "01Loss"
        self.modelconfigurefile = "modelconfigures/MultiMF01Loss_config"

    def update(self, instance):
        uid, iid, lid = instance
        ## calculate update step ##
        # intermediate #
        m = np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1)
        mgrad = -m
        mgrad[lid] = 1.0 + mgrad[lid]
        # for u #
        delt_u = np.transpose(np.multiply(np.transpose(self.v[iid]), mgrad))
        # for v #
        delt_v = np.transpose(np.multiply(np.transpose(self.u[uid]), mgrad))
        self.u[uid] += (self.SGDstep * delt_u)
        self.v[iid] += (self.SGDstep * delt_v)
        return self

    def loss(self, test):
        losssum = 0.0
        nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid = samp
            self.initialize(uid, iid, predict = True)
            m = np.sum(np.multiply(self.u[uid], self.v[iid]), axis = 1)
            m_true = np.zeros(self.L)
            m_true[lid] = 1.0
            losssum += np.sum(np.power((m - m_true), 2.0))
            nsamp += 1
        return losssum/nsamp