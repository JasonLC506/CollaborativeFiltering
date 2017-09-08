"""
Based on MultiMA Decomposition assumption
synthesizing multiclass dyadic data
"""

import numpy as np
import cPickle

SCALE = 5.0

class MultiMAsynthetic(object):
    def __init__(self, N, M, L, K):
        self.K = K
        self.N = N
        self.M = M
        self.L = L

        self.u = None
        self.v = None

        self.dyaddict = {}

        self.parameter()

        self.bayesian_error = []
        self.bayesian_entropy = []

    def parameter(self):
        self.u = np.random.normal(scale=SCALE, size = (self.N, self.L))
        self.v = np.random.normal(scale=SCALE, size = (self.M, self.L))
        return self

    def generate(self, fraction, MemorySampleSize=1e+7):
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
                    lid = self._singleGenerate(uid, iid, distribution=False)
                    yield [uid, iid, lid]
                    cnt += 1
        else:
            for uid in range(self.N):
                for iid in range(self.M):
                    lid = self._singleGenerate(uid, iid, distribution=False)
                    yield [uid, iid, lid]

    def generate2file(self, fraction, filename):
        with open(filename, "w") as f:
            for samp in self.generate(fraction):
                transaction = {"POSTID": samp[0],
                               "READERID": samp[1],
                               "EMOTICON": samp[2]}
                f.write(str(transaction) + "\n")
        return self

    def _singleGenerate(self, uid, iid, distribution=False):
        m = self.u[uid] + self.v[iid]
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            lid = np.argmax(np.random.multinomial(1, expm / expmsum, size=1))
            self.bayesian_error.append((np.sum(expm) - expm[lid]) / expmsum)  # calculate bayesian error
            self.bayesian_entropy.append(-np.log(expm[lid]/expmsum))
            return lid

    def modelPrint2File(self, filename):
        with open(filename, "w") as f:
            cPickle.dump({"u": self.u, "v": self.v}, f)

if __name__ == "__main__":
    np.random.seed(2017)
    N = 500
    M = 500
    L = 3

    K = 1
    generator = MultiMAsynthetic(N=N,M=M,L=L,K=K)
    generator.generate2file(1.0, "data/MultiMAsynthetic_N%d_M%d_L%d_K%d" % (N,M,L,K))
    bayesian_error = np.array(generator.bayesian_error)
    bayesian_entropy = np.array(generator.bayesian_entropy)
    print "bayesian error: ",  np.mean(bayesian_error), np.std(bayesian_error)
    print "bayesian entropy: ", np.mean(bayesian_entropy), np.std(bayesian_entropy)