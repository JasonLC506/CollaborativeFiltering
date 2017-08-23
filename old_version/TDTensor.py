"""
General tensor (3-way) TD
Same assumption, small and given for last dimension
"""

import numpy as np
from TDMultiClass import TD
import matrixTool

class TDTensor(TD):
    def update(self, instance):
        uid, iid, lid, value = instance
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r[lid])
        mgrad = 2.0 * (value - m)
        # gradient for embeddings #
        delt_u = mgrad * np.tensordot(a=self.v[iid], axes=(0, 1),
                                      b=np.tensordot(a=self.c, axes=(2, 0),
                                                     b=self.r[lid])
                                      )
        delt_v = mgrad * np.tensordot(a=self.u[uid], axes=(0, 0),
                                      b=np.tensordot(a=self.c, axes=(2, 0),
                                                     b=self.r[lid])
                                      )
        delt_c = mgrad * matrixTool.TensorOuter([self.u[uid], self.v[iid], self.r[lid]])
        delt_r = mgrad * np.tensordot(a=self.u[uid], axes=(0, 0),
                                      b=np.tensordot(a=self.v[iid], axes=(0, 1),
                                                     b=self.c))
        # update #
        self.u[uid] += (self.SGDstep * delt_u)
        self.v[iid] += (self.SGDstep * delt_v)
        self.c += (self.SGDstep * delt_c)
        self.r[lid] += (self.SGDstep * delt_r)

    def loss(self, test):
        losssum = 0.0
        nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid, value = samp
            pred_value = self.predict(uid, iid, lid)
            perf = np.power((value - pred_value), 2.0)
            losssum += perf
            nsamp += 1
        return losssum / nsamp

    def predict(self, uid, iid, lid):
        self.initialize(uid, iid, predict=True)
        return TDreconstruct(self.c, self.u[uid], self.v[iid], self.r[lid])


def TDreconstruct(c, u, v, t):
    ## calculate TD reconstruction ##
    m = np.tensordot(a = t, axes = (0, 0),
                     b = np.tensordot(a = u, axes = (0, 0),
                                    b = np.tensordot(a = v, axes = (0, 1),
                                                   b = c)
                                    )
                     )  # np.array([self.L,])
    return m