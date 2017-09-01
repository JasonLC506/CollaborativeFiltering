"""
based on Tucker Decomposition assumption
synthesizing multiclass dyadic data
"""

import numpy as np
from TD import TDreconstruct
import cPickle

SCALE = 0.75

class TDsynthetic(object):
    def __init__(self, N, M, L, ku, kv, kr):
        # data size #
        self.N = N
        self.M = M
        self.L = L

        # intrinsic latent size #
        self.ku = ku
        self.kv = kv
        self.kr = kr

        # model parameters #
        self.c = None
        self.u = None
        self.v = None
        self.r = None

        # intermediate storage #
        self.dyaddict = {}

        # model setup #
        self.parameter()

        # bayesian error #
        self.bayesian_error = []

    def parameter(self):
        self.u = np.random.normal(scale=SCALE, size = self.N * self.ku).reshape([self.N, self.ku])
        self.v = np.random.normal(scale=SCALE, size = self.M * self.kv).reshape([self.M, self.kv])
        self.r = np.random.normal(scale=SCALE, size = self.L * self.kr).reshape([self.L, self.kr])
        self.c = np.random.normal(scale=SCALE, size = self.ku * self.kv * self.kr).reshape([self.ku, self.kv, self.kr])
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
        m = TDreconstruct(self.c, self.u[uid], self.v[iid], self.r)
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            lid = np.argmax(np.random.multinomial(1, expm / expmsum, size=1))
            self.bayesian_error.append((np.sum(expm) - expm[lid])/expmsum) # calculate bayesian error
            return lid

    def modelPrint2File(self, filename):
        with open(filename, "w") as f:
            cPickle.dump({"u": self.u, "v": self.v, "r": self.r, "c": self.c}, f)

if __name__ == "__main__":
    np.random.seed(2017)
    N = 500
    M = 500
    L = 3

    ku = 20
    kv = 20
    kr = 10
    generator = TDsynthetic(N=N,M=M,L=L,ku=ku,kv=kv,kr=kr)
    generator.generate2file(1.0, "data/TDsynthetic_N%d_M%d_L%d_ku%d_kv%d_kr%d" % (N,M,L,ku,kv,kr))
    generator.modelPrint2File("data/TDsynthetic_model_N%d_M%d_L%d_ku%d_kv%d_kr%d" % (N,M,L,ku,kv,kr))
    bayesian_error = np.array(generator.bayesian_error)
    print np.mean(bayesian_error), np.std(bayesian_error)