import numpy as np
from trainingDataGenerator import datagenerator

def summary(data):
    user_dict = {}
    item_dict = {}
    for entry in data.sample(random=False):
        uid, iid, lid = entry
        user_dict.setdefault(uid, [0,0,0,0,0,0])
        user_dict[uid][lid] += 1
        item_dict.setdefault(iid, [0,0,0,0,0,0])
        item_dict[iid][lid] += 1
    return user_dict, item_dict

def stats(data):
    user_dict, item_dict = summary(data)
    results = [0,0]
    for i in range(2):
        if i == 0:
            dict = user_dict
        else:
            dict = item_dict
        Nsamp = 0
        Nreact = 0
        entropy = 0.0
        bayesian_error = 0.0
        for labels in dict.values():
            Nsamp += 1
            nreact = sum(labels)
            Nreact += nreact
            entropy += (Entropy(np.array(labels, dtype=np.float64))*nreact)
            bayesian_error += (nreact - max(labels))
        results[i] = [Nsamp, Nreact, entropy/Nreact, bayesian_error/Nreact]
    return results

def Entropy(array):
    array += 0.0001
    distribution = array / np.sum(array)
    return np.inner(distribution, -np.log(distribution))

if __name__ == "__main__":
    data = datagenerator("data/reaction_NYTWaPoWSJ_K10_0.7train")
    print stats(data)