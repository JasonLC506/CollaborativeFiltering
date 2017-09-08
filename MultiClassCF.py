"""
Superclass of multiclass collaborative filtering embedding methods with SGD optimization
"""

import numpy as np

class MCCF(object):
    def __init__(self):
        # model hyperparameters #
        self.hyperparameters = []   # k for TD, CD, MF models

        # model parameters #
        self.L = 0                  # size of rating label set

        self.paras_predefined = {}  # set of parameters whose values are initialized before data feeding
        self.paras_ascome = {}      # set of parameters whose values are initialized after new data feeding

        self.paras_ascome_default = {}  # default values for paras_ascome when model used for prediction

        # fitting hyperparameters #
        self.SCALE = 0.1
        self.SGDstep = 0.0001
        self.lamda = 0.000

        # results storage #
        self.logfilename = "MCCF"   ## model dependent
        self.modelconfigurefile = "MCCF_config"     ## model dependent

    def fit(self, training, valid, model_hyperparameters, max_epoch = 1000, SGDstep = 0.001, SCALE = 0.1, lamda = 0.001):
        # model independent #
        self.set_model_hyperparameters(model_hyperparameters)
        self.L = training.L()
        self.SGDstep = SGDstep
        self.SCALE = SCALE
        self.lamda = lamda

        with open(self.logfilename, "a") as logf:
            logf.write("model_hyperparameters: " + str(model_hyperparameters) + "\n")
            logf.write("SGDstep: " + str(self.SGDstep) + "\n")
            logf.write("SCALE: " + str(self.SCALE) + "\n")

        self.modelconfigurefile += str(model_hyperparameters) + "_SGDstep" + str(self.SGDstep) + "_SCALE" + str(self.SCALE)

        self.basicInitialize()

        # SGD #
        loss_valid = None
        loss_valid_minimum = None
        for epoch in xrange(max_epoch):
            for samp in training.sample():
                self.initialize(samp[0], samp[1])
                self.update(samp)
            self.averageEmbedding()
            loss_train_new = self.loss(training)
            loss_valid_new = self.loss(valid)
            with open(self.logfilename, "a") as logf:
                logf.write("after epoch %d loss training: %f\n" % (epoch, loss_train_new))
                logf.write("after epoch %d loss valid: %f\n" % (epoch, loss_valid_new))
            if loss_valid is not None and loss_valid_new >= loss_valid:
                with open(self.logfilename, "a") as logf:
                    logf.write("overfitting in epoch: %d\n" % epoch)
            else:
                if loss_valid_minimum is None or loss_valid_new < loss_valid_minimum:
                    loss_valid_minimum = loss_valid_new
                    self.modelconfigStore()
            loss_valid = loss_valid_new
        self.modelconfigStore(self.modelconfigurefile + "end_epoch" + str(max_epoch))
        return self

    def set_model_hyperparameters(self, model_hyperparameters):
        ## model dependent ##
        self.hyperparameters = model_hyperparameters
        print "set_model_hyperparameters not defined"

    def basicInitialize(self):
        ## model dependent ##
        self.paras_predefined = {}
        print "basicInitialize not defined"

    def initialize(self, uid, iid, predict = False):
        ## model dependent ##
        self.paras_ascome = {}
        self.paras_ascome_default = {}
        print "initialize not defined"

    def update(self, instance):
        ## model dependent ##
        print "update not defined"

    def averageEmbedding(self):
        ## model dependent ##
        """
        self.paras_ascome_default set as avg of self.paras_ascome
        """
        print "averageEmbedding not defined"

    def loss(self, test):
        ## model dependent ##
        """
        default as log likelihood loss
        """
        losssum = 0.0
        nsamp = 0
        for samp in test.sample(random = False):
            uid, iid, lid = samp
            predprob = self.predict(uid, iid, distribution = True)
            perf = ppl(predprob = predprob, truelabel = lid)
            losssum += perf
            nsamp += 1
        return losssum/nsamp

    def predict(self, uid, iid, distribution = True):
        ## model dependent ##
        self.initialize(uid, iid, predict = True)
        print "predict not defined"
        return []

    def modelconfigStore(self, modelconfigurefile = None):
        ## model dependent ##
        """
        store model config in self.modelconfigurefile
        """

    def modelconfigLoad(self, modelconfigurefile = None):
        ## model dependent ##
        """
        load model configuration from modelconfigurefile
        """


def ppl(predprob, truelabel):
    return (- np.log(predprob[truelabel]))