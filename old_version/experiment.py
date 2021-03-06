import numpy as np
# from MFMultiClass import MF
# from TDMultiClass import TD
# from CDMultiClass import CD
from PITF_old import PITF
# from TDMultiClass_square01 import TD
# from CDMultiClass_GradientDescent import CD
# from CDMultiClass_MFreduced import CD
# from CDMultiClass_square01 import CD
# from TDMultiClass_parameterCopy_Fix import TD
from trainingDataGenerator import datagenerator

# def experiment(data_train, data_valid, data_test, ku, kv, kr, max_epoch = 10, SGDstep = 0.001):
def experiment(data_train, data_valid, data_test, K, max_epoch=10, SGDstep=0.001):

    training = datagenerator(data_train)
    valid = datagenerator(data_valid)
    test = datagenerator(data_test)

    # model = MF()
    # model = CD()
    model = PITF()
    model.fit(training, valid, K = K, max_epoch = max_epoch, SGDstep = SGDstep)
    # model = TD()
    # model.fit(training, valid, ku, kv, kr, max_epoch=max_epoch, SGDstep=SGDstep)
    perf = performance(test, model)
    return perf


def performance(test, model):
    ppl = model.loss(test)
    # model prediction distribution#
    dist = {}
    # 0-1 accuracy#
    correct = 0
    Ntest = 0
    for samp in test.sample(random = False):
        uid, iid, lid = samp
        lid_pred = model.predict(uid, iid, distribution=False)
        # lid_pred = model.predict(uid, iid)
        dist.setdefault(lid_pred,[0,0])[0] += 1
        if lid == lid_pred:
            correct += 1
            dist[lid_pred][1] += 1
        Ntest += 1
    accuracy = 1.0 * correct / Ntest
    return [ppl, accuracy, dist]

if __name__ == "__main__":
    np.random.seed(2017)
    print experiment(data_train = "data/synthetic_N500_M500_L3_K5_0.7train",
                     data_valid = "data/synthetic_N500_M500_L3_K5_0.1valid",
                     data_test = "data/synthetic_N500_M500_L3_K5_0.2test",
                     # ku = 15, kv = 15, kr = 5,
                     K = 15,
                     max_epoch = 1000, SGDstep = 0.01)
    # print experiment(data_train="data/TDsynthetic_N500_M500_L3_ku20_kv20_kr10_0.7train",
    #                  data_valid="data/TDsynthetic_N500_M500_L3_ku20_kv20_kr10_0.1valid",
    #                  data_test="data/TDsynthetic_N500_M500_L3_ku20_kv20_kr10_0.2test",
    #                  ku=20, kv=20, kr=10,
    #                  # K = 100,
    #                  max_epoch=1000, SGDstep=0.01)