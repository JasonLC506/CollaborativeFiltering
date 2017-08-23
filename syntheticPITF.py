"""
based on PITF
synthesizing multiclass dyadic data
"""

import numpy as np
import cPickle

SCALE = 1.0

class PITFsynthetic(object):
    def __init__(self, N, M, L, K):
        # data size #
        self.N = N
        self.M = M
        self.L = L

        # intrinsic latent size #
        self.k = K

        # model parameters #
        self.u = None
        self.v = None
        self.r_u = None
        self.r_v = None

        # intermediate storage #
        self.dyaddict = {}

        # model setup #
        self.parameter()

        # bayesian error #
        self.bayesian_error = []

    def parameter(self):
        self.u = np.random.normal(scale=SCALE, size = self.N * self.k).reshape([self.N, self.k])
        self.v = np.random.normal(scale=SCALE, size = self.M * self.k).reshape([self.M, self.k])
        self.r_u = np.random.normal(scale=SCALE, size = self.L * self.k).reshape([self.L, self.k])
        self.r_v = np.random.normal(scale=SCALE, size=self.L * self.k).reshape([self.L, self.k])
        return self

    def generate(self, fraction, MemorySampleSize = 1e+7):
        Nsamp = self.N * self.M * fraction

        # deal with large sample size #
        if Nsamp >= MemorySampleSize:
            Mseg = int(MemorySampleSize / fraction / self.N)
        else:
            Mseg = self.M

        cnt = 0
        seg = 0
        if fraction < 1.0:
            while cnt < Nsamp:
                # stage-wise sampling for large sampling size #
                if cnt < MemorySampleSize * (seg + 1):
                    iid = np.random.randint(Mseg * seg, min([Mseg * (seg + 1), self.M]))
                else:
                    seg += 1
                    self.dyaddict.clear()
                    print seg, len(self.dyaddict)
                    continue
                uid = np.random.randint(0, self.N)
                dyadid = uid + iid * self.N
                if dyadid in self.dyaddict:
                    continue
                else:
                    self.dyaddict[dyadid] = cnt
                    lid = self._singleGenerate(uid, iid, distribution = False)
                    yield [uid, iid, lid]
                    cnt += 1
        else:
            for uid in range(self.N):
                for iid in range(self.M):
                    lid = self._singleGenerate(uid, iid, distribution = False)
                    yield [uid, iid, lid]

    def generate2file(self, fraction, filename):
        with open(filename, "w") as f:
            for samp in self.generate(fraction):
                transaction = {"POSTID": samp[0],
                               "READERID": samp[1],
                               "EMOTICON": samp[2]}
                f.write(str(transaction) + "\n")
        return self

    def _singleGenerate(self, uid, iid, distribution = False):
        m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
        return np.argmax(m)

    def modelPrint2File(self, filename):
        with open(filename, "w") as f:
            cPickle.dump({"u": self.u, "v": self.v, "r_u": self.r_u, "r_v": self.r_v}, f)

if __name__ == "__main__":
    np.random.seed(2017)
    N = 500
    M = 500
    L = 3
    K = 15
    generator = PITFsynthetic(N=N,M=M,L=L,K = K)
    generator.generate2file(1.0, "data/PITFsynthetic_N%d_M%d_L%d_K%d" % (N,M,L,K))
    generator.modelPrint2File("data/PITFsynthetic_model_N%d_M%d_L%d_K%d" % (N,M,L,K))
    bayesian_error = np.array(generator.bayesian_error)
    print np.mean(bayesian_error), np.std(bayesian_error)