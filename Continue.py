from trainingDataGenerator import datagenerator
from BiNN_separate_single import BiNNSeparateSingle
import cPickle

SCALE = 0.1

datafile = "data/reaction_NYTWaPoWSJ_K10_"
data_train = datagenerator(datafile + "0.7train")
data_valid = datagenerator(datafile + "0.1valid")

model = BiNNSeparateSingle()
model.modelconfigLoad(modelconfigurefile="modelconfigures/BiNNSeparateSingle_config_reaction_NYTWaPoWSJ_K10_0.7train[6]_SGDstep0.01_SCALE0.1_lamda0.001")
model.fit(data_train, data_valid, model_hyperparameters=[6], max_epoch=100, SGDstep=0.01, SCALE=SCALE, lamda=0.0,
             no_initialization=True)