"""
pairwise interaction tensor factorization (PITF) model,
pairwise comapison among tags and pairwise decomposition of tensor by user-tag and item-tag additively, based on
Rendle, S. and Schmidt-Thieme, L., 2010, February. Pairwise interaction tensor factorization for personalized tag recommendation. In Proceedings of the third ACM international conference on Web search and data mining (pp. 81-90). ACM. [1]
"""
import numpy as np
from CDMultiClass import CD

SCALE = 0.01

class PITF(CD):
    def __init__(self):
        CD.__init__(self)
        self.r_u = None
        self.r_v = None
        self.lamda = 0.000
        self.SCALE = SCALE

    def basicInitialize(self):
        self.r_u = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))
        self.r_v = np.random.normal(0.0, self.SCALE, size = (self.L, self.k))

    def update(self, instance, isamp):
        uid, iid, lid = instance
        lid_neg_list = [i for i in range(self.L)]
        del lid_neg_list[lid]

        m = np.tensordot(self.r_u, self.u[uid], axes = (1,0)) + np.tensordot(self.r_v, self.v[iid], axes = (1,0))
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
        return self

    def loss(self, test):
        losssum = 0.0
        Nsamp = 0
        for samp in test.sample(random=False):
            losssum += self.lossSingle(samp)
            Nsamp += 1
        for uid in self.u.keys():
            losssum = losssum + self.lamda * np.power(np.linalg.norm(self.u[uid]),2)
        for iid in self.v.keys():
            losssum = losssum + self.lamda * np.power(np.linalg.norm(self.v[iid]),2)
        losssum = losssum + self.lamda * (np.power(np.linalg.norm(self.r_u), 2) + np.power(np.linalg.norm(self.r_v), 2))
        return losssum / Nsamp

    def lossSingle(self, instance):
        uid, iid, lid = instance
        self.initialize(uid,iid,predict=True)
        m = np.tensordot(self.r_u, self.u[uid], axes = (1,0)) + np.tensordot(self.r_v, self.v[iid], axes = (1,0))
        lid_neg_list = [i for i in range(self.L)]
        del lid_neg_list[lid]
        loss = 0.0
        for lid_neg in lid_neg_list:
            loss += np.log(sigmoid(m[lid] - m[lid_neg]))
        loss = - loss
        return loss

    def predict(self, uid, iid, distribution = False):
        self.initialize(uid, iid, predict=True)
        m = np.tensordot(self.r_u, self.u[uid], axes = (1,0)) + np.tensordot(self.r_v, self.v[iid], axes = (1,0))
        return np.argmax(m)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))