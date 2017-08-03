from MFMultiClass import MF
from TDMultiClass import TD
from trainingDataGenerator import datagenerator

def experiment(data_train, data_valid, data_test, ku, kv, kr, max_epoch = 10, SGDstep = 0.001):
# def experiment(data_train, data_valid, data_test, K, max_epoch=10, SGDstep=0.001):

    training = datagenerator(data_train)
    valid = datagenerator(data_valid)
    test = datagenerator(data_test)

    # model = MF()
    # model.fit(training, valid, K = K, max_epoch = max_epoch, SGDstep = SGDstep)
    model = TD()
    model.fit(training, valid, ku, kv, kr, max_epoch=max_epoch, SGDstep=SGDstep)
    perf = performance(test, model)
    return perf


def performance(test, model):
    ppl = model.loss(test)
    # 0-1 accuracy#
    correct = 0
    Ntest = 0
    for samp in test.sample(random = False):
        uid, iid, lid = samp
        lid_pred = model.predict(uid, iid, distribution=False)
        if lid == lid_pred:
            correct += 1
        Ntest += 1
    accuracy = 1.0 * correct / Ntest
    return [ppl, accuracy]

if __name__ == "__main__":
    print experiment(data_train = "data/synthetic_N500_M500_L3_K5_0.7train",
                     data_valid = "data/synthetic_N500_M500_L3_K5_0.1valid",
                     data_test = "data/synthetic_N500_M500_L3_K5_0.2test",
                     ku = 15, kv = 15, kr = 2,
                     # K = 5,
                     max_epoch = 1000, SGDstep = 0.001)