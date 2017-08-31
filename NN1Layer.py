"""
Inspired by the PITF model, we change the loss function into loglikelihood with softmax in multiclassCF context,
which is then equivalent to Neural Network with 1 layer (linear model)
"""

import numpy as np
import cPickle
from MultiClassCF import MCCF
from MultiClassCF import ppl

class NN1Layer(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        # instantiate hyperparameters #
        self.k = 0

        # paras predefined #
        self.r_u = None
        self.r_v = None

        # paras ascome #
        self.u = {}
        self.v = {}

        # paras ascome default #
        self.u_avg = None
        self.v_avg = None

        self.lamda = 0.0

        # results storage #
        self.logfilename = "logs/NN1Layer_GD"
        self.modelconfigurefile = "modelconfigures/NN1Layer_GD_config"

        ### test ###
        self.delt_r_u = None
        self.delt_r_v = None
        self.delt_u = {}
        self.delt_v = {}

    def fit(self, training, valid, model_hyperparameters, max_epoch=1000, SGDstep=0.001, SCALE=0.1):
        # model independent #
        self.set_model_hyperparameters(model_hyperparameters)
        self.L = training.L()
        self.SGDstep = SGDstep
        self.SCALE = SCALE

        with open(self.logfilename, "a") as logf:
            logf.write("model_hyperparameters: " + str(model_hyperparameters) + "\n")
            logf.write("SGDstep: " + str(self.SGDstep) + "\n")
            logf.write("SCALE: " + str(self.SCALE) + "\n")

        self.modelconfigurefile += str(model_hyperparameters) + "_SGDstep" + str(self.SGDstep) + "_SCALE" + str(
            self.SCALE)

        self.basicInitialize()

        ### test ###
        self.modelconfigLoad("modelconfigures/NN1Layer_config_PITFsynthetic_N500_M500_L3_K15_0.7train[15]_SGDstep0.01_SCALE0.1")
        for key in self.u.keys():
            self.delt_u[key] = np.zeros(self.u[key].shape, dtype=np.float64)
        for key in self.v.keys():
            self.delt_v[key] = np.zeros(self.v[key].shape, dtype=np.float64)

        # SGD #
        loss_valid = None
        loss_valid_minimum = None
        for epoch in xrange(max_epoch):
            loss_train_old = self.loss(training)
            loss_valid_old = self.loss(valid)
            Nsamp = 0
            for samp in training.sample():
                self.initialize(samp[0], samp[1])
                self.stepupdate(samp)
                Nsamp += 1
            self.update(Nsamp)
            self.averageEmbedding()
            loss_train_new = self.loss(training)
            loss_valid_new = self.loss(valid)
            print "before epoch, train", loss_train_old
            print "after epoch, train", loss_train_new
            print "before epoch, valid", loss_valid_old
            print "after epoch, valid", loss_valid_new
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
        if len(model_hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = model_hyperparameters[0]

    def basicInitialize(self):
        self.r_u = np.random.normal(0.0, self.SCALE, size=(self.L, self.k))
        self.r_v = np.random.normal(0.0, self.SCALE, size=(self.L, self.k))
        self.delt_r_u = np.zeros(self.r_u.shape, dtype=np.float64)
        self.delt_r_v = np.zeros(self.r_v.shape, dtype=np.float64)
        self.u_avg = np.zeros(self.k)
        self.v_avg = np.zeros(self.k)

    def initialize(self, uid, iid, predict=False):
        ## according to [1] in MultiMF ##
        if uid not in self.u:
            if predict:
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, self.SCALE, size=self.k)
            self.delt_u[uid] = np.zeros(self.u[uid].shape, dtype=np.float64)
        if iid not in self.v:
            if predict:
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, self.SCALE, size=self.k)
            self.delt_v[iid] = np.zeros(self.v[iid].shape, dtype=np.float64)
        return self

    # def update(self, instance):
    #     loss_before = self.singleLoss(instance)
    #     uid, iid, lid = instance
    #     m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
    #     expm = np.exp(m)
    #     expmsum = np.sum(expm)
    #     mgrad = expm / expmsum
    #     mgrad[lid] = mgrad[lid] - 1.0
    #     mgrad = - mgrad
    #     # gradient for embeddings #
    #     delt_u = np.tensordot(mgrad, self.r_u, axes = (0,0))
    #     delt_r_u = np.outer(mgrad, self.u[uid])
    #     delt_v = np.tensordot(mgrad, self.r_v, axes = (0,0))
    #     delt_r_v = np.outer(mgrad, self.v[iid])
    #     # update #
    #     self.u[uid] += (self.SGDstep * delt_u)
    #     self.v[iid] += (self.SGDstep * delt_v)
    #     self.r_u += (self.SGDstep * delt_r_u)
    #     self.r_v += (self.SGDstep * delt_r_v)
    #     loss_after = self.singleLoss(instance)
    #     if loss_after > loss_before:
    #         print "single loss increase", loss_before, loss_after
    #     return self

    def stepupdate(self, instance):
        uid, iid, lid = instance
        m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        mgrad = expm / expmsum
        mgrad[lid] = mgrad[lid] - 1.0
        mgrad = - mgrad
        # gradient for embeddings #
        delt_u = np.tensordot(mgrad, self.r_u, axes = (0,0))
        delt_r_u = np.outer(mgrad, self.u[uid])
        delt_v = np.tensordot(mgrad, self.r_v, axes = (0,0))
        delt_r_v = np.outer(mgrad, self.v[iid])
        # update #
        self.delt_u[uid] += delt_u
        self.delt_v[iid] += delt_v
        self.delt_r_u += delt_r_u
        self.delt_r_v += delt_r_v
        return self

    def update(self, Nsamp):
        for uid in self.delt_u.keys():
            self.u[uid] += (self.SGDstep * self.delt_u[uid]/Nsamp)
            del self.delt_u[uid]
            self.delt_u[uid] = np.zeros(self.u[uid].shape, dtype=np.float64)
        for iid in self.delt_v.keys():
            self.v[iid] += (self.SGDstep * self.delt_v[iid]/Nsamp)
            del self.delt_v[iid]
            self.delt_v[iid] = np.zeros(self.v[iid].shape, dtype=np.float64)
        self.r_u += (self.SGDstep * self.delt_r_u/Nsamp)
        del self.delt_r_u
        self.delt_r_u = np.zeros(self.r_u.shape, dtype=np.float64)
        self.r_v += (self.SGDstep * self.delt_r_v/Nsamp)
        del self.delt_r_v
        self.delt_r_v = np.zeros(self.r_v.shape, dtype=np.float64)

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis=0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis=0)

    def predict(self, uid, iid, distribution = True):
        self.initialize(uid, iid, predict = True)
        m = np.tensordot(self.r_u, self.u[uid], axes=(1, 0)) + np.tensordot(self.r_v, self.v[iid], axes=(1, 0))
        expm = np.exp(m)
        expmsum = np.sum(expm)
        if distribution:
            return expm / expmsum
        else:
            return np.argmax(expm)

    ### test ###
    def singleLoss(self, instance):
        predprod = self.predict(instance[0], instance[1], distribution=True)
        loss = ppl(predprob=predprod, truelabel=instance[2])
        return loss

    def modelconfigStore(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {"u": self.u, "v": self.v, "r_u": self.r_u, "r_v": self.r_v, "u_avg": self.u_avg, "v_avg": self.v_avg}
        with open(modelconfigurefile, "w") as f:
            cPickle.dump(paras, f)

    def modelconfigLoad(self, modelconfigurefile=None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        with open(modelconfigurefile, "r") as f:
            paras = cPickle.load(f)
        # write to model parameters #
        self.u = paras["u"]
        self.v = paras["v"]
        self.r_u = paras["r_u"]
        self.r_v = paras["r_v"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.r_u.shape[0]
        print "model successfully loaded from " + modelconfigurefile