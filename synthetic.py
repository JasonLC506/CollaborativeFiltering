"""
based on [1] in MFMultiClass
"""
import numpy as np

class MFsynthetic(object):
    def __init__(self, N, M, L, K):
        self.N = N
        self.M = M
        self.L = L
        self.K = K

        self.u = None
        self.v = None
        self.parameter()
        self.dyaddict = {}

    def parameter(self):
        self.u = np.random.random(self.N*self.L*self.K).reshape([self.N, self.L, self.K]) * 6.0 - 3.0
        self.v = np.random.random(self.M*self.L*self.K).reshape([self.M, self.L, self.K]) * 6.0 - 3.0

    def generate(self, fraction, MemorySampleSize = 1e+8):
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
                    iid = np.random.randint(Mseg * seg, Mseg * (seg + 1))
                else:
                    seg += 1
                    self.dyaddict.clear()
                    continue
                uid = np.random.randint(0, self.N)
                dyadid = uid + iid * self.N
                if dyadid in self.dyaddict:
                    continue
                else:
                    self.dyaddict[dyadid] = cnt
                    expprod = np.exp(np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1))
                    expprodsum = np.sum(expprod)
                    lid = np.argmax(np.random.multinomial(1, expprod/expprodsum, size = 1))
                    yield [uid, iid, lid]
                    cnt += 1
        else:
            for uid in range(self.N):
                for iid in range(self.M):
                    expprod = np.exp(np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1))
                    expprodsum = np.sum(expprod)
                    lid = np.argmax(np.random.multinomial(1, expprod/expprodsum, size = 1))
                    yield [uid, iid, lid]

    def generate2file(self, fraction, filename):
        with open(filename, "w") as f:
            for samp in self.generate(fraction):
                transaction = {"POSTID": samp[0],
                               "READERID": samp[1],
                               "EMOTICON": samp[2]}
                f.write(str(transaction) + "\n")
        return self


if __name__ == "__main__":
    np.random.seed(2017)
    N = 20000
    M = 10000000
    L = 3
    K = 5
    generator = MFsynthetic(N = N, M = M, L = L, K = K) # based on [1]
    generator.generate2file(0.002, "data/synthetic_N%d_M%d_L%d_K%d" % (N,M,L,K))