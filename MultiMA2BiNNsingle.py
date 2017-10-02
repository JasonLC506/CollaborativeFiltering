import numpy as np
from BiNN_single import BiNNsingle
from MultiMA import MultiMA
import copy
from trainingDataGenerator import datagenerator

model_MA = MultiMA()
model_MA.modelconfigLoad(modelconfigurefile="modelconfigures/MultiMA_config_reaction_NYTWaPoWSJ_K10_0.7train[1]_SGDstep0.01_SCALE0.1")
model_BN = BiNNsingle()
model_BN.u = copy.deepcopy(model_MA.u)
model_BN.u_avg = copy.deepcopy(model_MA.u_avg)
model_BN.v = copy.deepcopy(model_MA.v)
model_BN.v_avg = copy.deepcopy(model_MA.v_avg)
model_BN.d = 6
model_BN.L = 6
model_BN.W1bi = np.zeros([model_BN.d,model_BN.d,model_BN.d])
model_BN.W1u = np.identity(model_BN.d, dtype = np.float64)
model_BN.W1v = np.identity(model_BN.d, dtype=np.float64)
model_BN.B1 = np.zeros(model_BN.d)

datafile = "data/reaction_NYTWaPoWSJ_K10_"
data_train = datagenerator(datafile + "0.7train")
data_valid = datagenerator(datafile + "0.1valid")
# data_test = datagenerator(datafile + "0.2test")
# print "final valid loss from MA", model_MA.loss(data_valid)
# print "final training loss from MA", model_MA.loss(data_train)
# print "initial valid loss from BN", model_BN.loss(data_valid)
# print "initial training loss from BN", model_BN.loss(data_train)

model_BN.modelconfigurefile += "load_from_MA"
model_BN.logfilename += "load_from_MA"
model_BN.fit(data_train, data_valid, model_hyperparameters=[6], max_epoch = 100, SGDstep = 0.001, SCALE = 0.1, lamda = 0.00,
             no_initialization = True)
