"""
generate Tensor with Tucker Decomposition assumption
"""

import numpy as np
from syntheticTD import TDsynthetic


class TDTsynthetic(TDsynthetic):
    def __init__(self, N, M, L, ku, kv, kr):
        TDsynthetic.__init__(self, N, M, L, ku, kv, kr)
        self.ternarydict = {}

    def generate(self, fraction, MemorySampleSize = 1e+7):
        Nsamp = self.N * self.M * self.L * fraction

        # deal with large sample size #
        if Nsamp >= MemorySampleSize:
            Mseg = int(MemorySampleSize / fraction / self.N / self.L)
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
                lid = np.random.randint(0, self.L)
                tripleid = uid * self.L + iid * self.N * self.L + lid
                if tripleid in self.ternarydict:
                    continue
                else:
                    self.dyaddict[tripleid] = cnt
                    value = np.asscalar(TDreconstruct(self.c, self.u[uid], self.v[iid], self.r[lid]))
                    yield [uid, iid, lid, value]
                    cnt += 1
        else:
            for uid in range(self.N):
                for iid in range(self.M):
                    for lid in range(self.L):
                        value = np.asscalar(TDreconstruct(self.c, self.u[uid], self.v[iid], self.r[lid]))
                        yield [uid, iid, lid, value]

    def generate2file(self, fraction, filename):
        with open(filename, "w") as f:
            for samp in self.generate(fraction):
                f.write(str(samp) + "\n")
        return self

def TDreconstruct(c, u, v, t):
    ## calculate TD reconstruction ##
    m = np.tensordot(a = t, axes = (0, 0),
                     b = np.tensordot(a = u, axes = (0, 0),
                                    b = np.tensordot(a = v, axes = (0, 1),
                                                   b = c)
                                    )
                     )  # np.array([self.L,])
    return m

if __name__ == "__main__":
    np.random.seed(2017)
    N = 500
    M = 500
    L = 3
    ku = 20
    kv = 20
    kr = 10
    generator = TDTsynthetic(N=N,M=M,L=L,ku=ku,kv=kv,kr=kr)
    generator.generate2file(1.0, "data/TDTsynthetic_N%d_M%d_L%d_ku%d_kv%d_kr%d" % (N,M,L,ku,kv,kr))
    generator.modelPrint()