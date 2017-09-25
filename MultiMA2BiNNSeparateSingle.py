import numpy as np
from BiNN_single import BiNNsingle
from BiNN_separate_single import BiNNSeparateSingle
from MultiMA import MultiMA
import copy
from trainingDataGenerator import datagenerator

def dictionary_copy2part(dict_source, default_part_dim):
    default_part = np.zeros([default_part_dim])
    dict_new = {}
    for key in dict_source.keys():
        dict_new[key] = [copy.deepcopy(default_part), dict_source[key]]
    return dict_new

L = 6
d = 6

model_MA = MultiMA()
model_MA.modelconfigLoad(modelconfigurefile="modelconfigures/MultiMA_config_reaction_NYTWaPoWSJ_K10_0.7train[1]_SGDstep0.01_SCALE0.1")
model_BN = BiNNSeparateSingle()
model_BN.u = dictionary_copy2part(model_MA.u,d)
model_BN.u_avg = [np.zeros([d]), model_MA.u_avg]
model_BN.v = dictionary_copy2part(model_MA.v,d)
model_BN.v_avg = [np.zeros([d]), model_MA.v_avg]
model_BN.d = d
model_BN.L = L
model_BN.W1bi = np.zeros([model_BN.L,model_BN.d,model_BN.d])
model_BN.B1 = np.zeros(model_BN.L)

datafile = "data/reaction_NYTWaPoWSJ_K10_"
data_train = datagenerator(datafile + "0.7train")
data_valid = datagenerator(datafile + "0.1valid")
data_test = datagenerator(datafile + "0.2test")
print "final valid loss from MA", model_MA.loss(data_valid)
print "final training loss from MA", model_MA.loss(data_train)
print "initial valid loss from BN", model_BN.loss(data_valid)
print "initial training loss from BN", model_BN.loss(data_train)

model_BN.modelconfigurefile += "load_from_MA"
model_BN.logfilename += "load_from_MA"
model_BN.fit(data_train, data_valid, model_hyperparameters=[6], max_epoch = 100, SGDstep = 0.001, SCALE = 0.1, lamda = 0.01,
             no_initialization = True)
