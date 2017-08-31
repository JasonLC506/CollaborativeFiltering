import numpy as np
from trainingDataGenerator import datagenerator
from MultiMF import MultiMF
from TD import TD
from CD import CD
from TD01Loss import TD01Loss
from CD01Loss import CD01Loss
from MultiMF01Loss import MultiMF01Loss
from PITF import PITF
from NN1Layer import NN1Layer
from NTN import NTN


def fit(data_train, data_valid, method, hyperparameters, max_epoch=10, SGDstep=0.001, SCALE = 0.1):

    training = datagenerator(data_train)
    valid = datagenerator(data_valid)

    model = method()
    model.logfilename += "_" + data_train[5:]
    model.modelconfigurefile += "_" + data_train[5:]
    model.fit(training, valid, model_hyperparameters = hyperparameters, max_epoch = max_epoch, SGDstep = SGDstep)
    return model.modelconfigurefile


def performance(data_test, method, modelconfigurefile):
    test = datagenerator(data_test)
    model = method()
    model.modelconfigLoad(modelconfigurefile)
    model_loss = model.loss(test)
    # model prediction distribution#
    dist = {}
    # 0-1 accuracy#
    correct = 0
    Ntest = 0
    for samp in test.sample(random = False):
        uid, iid, lid = samp
        lid_pred = model.predict(uid, iid, distribution=False)
        # lid_pred = model.predict(uid, iid)
        dist.setdefault(lid,[0,0])[0] += 1
        if lid == lid_pred:
            correct += 1
            dist[lid][1] += 1
        Ntest += 1
    accuracy = 1.0 * correct / Ntest
    return [model_loss, accuracy]


def experiement(data_train, data_valid, data_test, method, hyperparameters, max_epoch=10, SGDstep=0.001, SCALE = 0.1):
    modelconfigurefile = fit(data_train, data_valid, method, hyperparameters, max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE)
    return performance(data_test, method, modelconfigurefile)


if __name__ == "__main__":
    np.random.seed(2017)
    datafile = "data/TDsynthetic_N500_M500_L3_ku20_kv20_kr10"
    data_train = datafile + "_0.7train"
    data_valid = datafile + "_0.1valid"
    data_test = datafile + "_0.2test"
    max_epoch = 500
    SGDstep = 0.01
    SCALE = 0.1

    # methods_list = [MultiMF, TD, CD]
    # method_names = ["MultiMF", "TD", "CD"]
    # hyperparameters_list_list = [[[3], [5], [15]],
    #                         [[9,9,3], [15,15,5], [45,45,15]],
    #                         [[9], [15], [45]]]
    method_names = ["NTN"]
    methods_list = [NTN]
    hyperparameters_list_list =[[[5,20]]]

    # method = NN1Layer
    # hyperparameters = [15]

    for i in range(len(methods_list)):
        method = methods_list[i]
        method_name = method_names[i]
        hyperparameters_list = hyperparameters_list_list[i]
        for hyperparameters in hyperparameters_list:
            print method_name
            print "hyperparameters =", hyperparameters

            # only fit model ##
            # print fit(data_train = data_train,
            #           data_valid = data_valid,
            #           method = method,
            #           hyperparameters = hyperparameters,
            #           max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE)

            ## only check model performance ##

            print performance(data_test = data_test,
                              method = method,
            #                   modelconfigurefile="modelconfigures/" + method_name +"_config_TDsynthetic_N500_M500_L3_ku15_kv15_kr5_0.7train" + str(hyperparameters) +"_SGDstep0.01_SCALE0.1")
                              modelconfigurefile = "modelconfigures/NTN_config_TDsynthetic_N500_M500_L3_ku20_kv20_kr10_0.7train[5, 20]_SGDstep0.01_SCALE0.1")

            ## fit & performance check ##
            # print experiement(data_train = data_train,
            #                   data_valid = data_valid,
            #                   data_test = data_test,
            #                   method = method,
            #                   hyperparameters = hyperparameters,
            #                   max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE)