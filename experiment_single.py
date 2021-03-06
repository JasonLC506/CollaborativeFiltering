import numpy as np
from trainingDataGenerator import datagenerator
from GMean import Gmeans
from MultiMF import MultiMF
from TD import TD
from CD import CD
from TD01Loss import TD01Loss
from CD01Loss import CD01Loss
from MultiMF01Loss import MultiMF01Loss
from MultiMA import MultiMA
from NTN import NTN
from BiNN import BiNN
from BiNN_single import BiNNsingle
#from PITF import PITF
#from NN1Layer import NN1Layer
import sys
import ast

def fit(data_train, data_valid, method, hyperparameters, max_epoch=10, SGDstep=0.001, SCALE = 0.1, lamda = 0.001):

    training = datagenerator(data_train)
    valid = datagenerator(data_valid)

    model = method()
    model.logfilename += "_" + data_train[5:]
    model.modelconfigurefile += "_" + data_train[5:]
    model.fit(training, valid, model_hyperparameters = hyperparameters, max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE, lamda = lamda)
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
            dist[lid_pred][1] += 1
        Ntest += 1
    accuracy = 1.0 * correct / Ntest
    print dist
    gmean = Gmeans(dist)
    return [model_loss, accuracy, gmean]


def experiement(data_train, data_valid, data_test, method, hyperparameters, max_epoch=10, SGDstep=0.001, SCALE = 0.1, lamda = 0.001):
    modelconfigurefile = fit(data_train, data_valid, method, hyperparameters, max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE, lamda = lamda)
    return performance(data_test, method, modelconfigurefile)


if __name__ == "__main__":
    np.random.seed(2017)
    datafile = "data/reaction_NYTWaPoWSJ_K10"
    data_train = datafile + "_0.7train"
    data_valid = datafile + "_0.1valid"
    data_test = datafile + "_0.2test"
    max_epoch = 1000
    SGDstep = 0.01
    SCALE = 0.1
    lamda = 0.001
   
    #SGDstep = float(ast.literal_eval(sys.argv[2]))
    
    result_file = "results/temporary.txt"

    # methods_list = [MultiMF, TD, CD]
    # method_names = ["MultiMF", "TD", "CD"]
    # hyperparameters_list_list = [[[3], [5], [15]],
    #                         [[9,9,3], [15,15,5], [45,45,15]],
    #                         [[9], [15], [45]]]
    method_names = ["BiNNsingle"]
    methods_list = [BiNNsingle]
    hyperparameters_list_list = [[[6]]]

    # method = NN1Layer
    # hyperparameters = [15]

    for i in range(len(methods_list)):
        method = methods_list[i]
        method_name = method_names[i]
        hyperparameters_list = hyperparameters_list_list[i]
        for hyperparameters in hyperparameters_list:

            # only fit model ##
            # print fit(data_train = data_train,
            #           data_valid = data_valid,
            #           method = method,
            #           hyperparameters = hyperparameters,
            #           max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE, lamda = lamda)

            ## only check model performance ##


            print performance(data_test = data_test,
                              method = method,
                              modelconfigurefile="modelconfigures/BiNNsingle_config_reaction_NYTWaPoWSJ_K10_0.7train[6]_SGDstep0.01_SCALE0.1")
                              # modelconfigurefile = "modelconfigures/NN1Layer_config_TDsynthetic_N500_M500_L3_K15_0.7train[15]_SGDstep0.01_SCALE0.1")

            ## fit & performance check ##
            # result = experiement(data_train = data_train,
            #                   data_valid = data_valid,
            #                   data_test = data_test,
            #                   method = method,
            #                   hyperparameters = hyperparameters,
            #                   max_epoch = max_epoch, SGDstep = SGDstep, SCALE = SCALE, lamda = lamda)

