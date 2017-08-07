from TDTensor import TDTensor
from trainingDataGeneratorTensor import datageneratorTensor as datagenerator

def experiment(data_train, data_valid, data_test, ku, kv, kr, max_epoch = 10, SGDstep = 0.001):
# def experiment(data_train, data_valid, data_test, K, max_epoch=10, SGDstep=0.001):

    training = datagenerator(data_train)
    valid = datagenerator(data_valid)
    test = datagenerator(data_test)

    # model = MF()
    # model = CD()
    # model.fit(training, valid, K = K, max_epoch = max_epoch, SGDstep = SGDstep)
    model = TDTensor()
    model.fit(training, valid, ku, kv, kr, max_epoch=max_epoch, SGDstep=SGDstep)
    perf = performance(test, model)
    return perf

def performance(test, model):
    mse = model.loss(test)
    return mse

if __name__ == "__main__":
    # print experiment(data_train = "data/synthetic_N500_M500_L3_K5_0.7train",
    #                  data_valid = "data/synthetic_N500_M500_L3_K5_0.1valid",
    #                  data_test = "data/synthetic_N500_M500_L3_K5_0.2test",
    #                  ku = 15, kv = 15, kr = 15,
    #                  # K = 15,
    #                  max_epoch = 1000, SGDstep = 0.001)
    print experiment(data_train="data/TDTsynthetic_N500_M500_L3_ku20_kv20_kr10_0.7train",
                     data_valid="data/TDTsynthetic_N500_M500_L3_ku20_kv20_kr10_0.1valid",
                     data_test="data/TDTsynthetic_N500_M500_L3_ku20_kv20_kr10_0.2test",
                     ku=20, kv=20, kr=10,
                     # K = 100,
                     max_epoch=1000, SGDstep=0.001)