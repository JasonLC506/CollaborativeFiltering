import numpy as np
from trainingDataGenerator import datagenerator

def summary(data):
    L = data.L()
    class_distribution = np.zeros(L)
    for samp in data.sample(random = False):
        uid, iid, lid= samp
        class_distribution[lid] += 1
    print class_distribution
    return class_distribution

if __name__ == "__main__":
    data = datagenerator("data/synthetic_N500_M500_L3_K5")
    summary(data)