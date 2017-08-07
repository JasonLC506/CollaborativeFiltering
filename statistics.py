import numpy as np
from trainingDataGeneratorTensor import datageneratorTensor

def summary(data):
    L = data.L()
    class_distribution = np.zeros(L)
    for samp in data.sample(random = False):
        uid, iid, lid, value = samp
        class_distribution[lid] += 1
    print class_distribution
    return class_distribution

if __name__ == "__main__":
    data = datageneratorTensor("data/TDTsynthetic_N500_M500_L3_ku20_kv20_kr10")
    summary(data)