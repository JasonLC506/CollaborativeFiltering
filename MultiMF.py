"""
implement multiclass matrix factorization algorithms for CF data using Stochastic Gradient Descent according to
Menon, A.K. and Elkan, C., 2010, December. A log-linear model with latent features for dyadic prediction. In Data Mining (ICDM), 2010 IEEE 10th International Conference on (pp. 364-373). IEEE. [1]
"""

import numpy as np
import cPickle
from MultiClassCF import MCCF, softmaxOutput, softmaxGradient


class MultiMF(MCCF):
    def __init__(self):
        MCCF.__init__(self)
        self.K = 0  # instantiate hyperparameters

        self.u = {}
        self.v = {}

        self.u_avg = None
        self.v_avg = None

        # results storage #
        self.logfilename = "logs/MultiMF"
        self.modelconfigurefile = "modelconfigures/MultiMF_config"

    def set_model_hyperparameters(self, model_hyperparameters):
        if len(model_hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.K = model_hyperparameters[0]

    def basicInitialize(self):
        print "No predefined parameters in MultiMF model"

    def initialize(self, uid, iid, predict = False):
        ## according to [1] ##
        if uid not in self.u:
            if predict:
                print "user new", uid###test
                self.u[uid] = self.u_avg
            else:
                self.u[uid] = np.random.normal(0.0, self.SCALE, size=self.L * self.K).reshape([self.L, self.K])
        if iid not in self.v:
            if predict:
                print "item new", iid###test
                self.v[iid] = self.v_avg
            else:
                self.v[iid] = np.random.normal(0.0, self.SCALE, size=self.L * self.K).reshape([self.L, self.K])
        return self

    def update(self, instance):
        """
        update embeddings according to a single instance with SGD
        instance: [UserId, ItemId, LabelId]
        """
        uid, iid, lid = instance
        ## calculate update step ##
        # intermediate #
        m = np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1)
        mgrad = softmaxGradient(m, lid)
        # for u #
        delt_u = np.transpose(np.multiply(np.transpose(self.v[iid]), mgrad))
        # for v #
        delt_v = np.transpose(np.multiply(np.transpose(self.u[uid]), mgrad))

        self.u[uid] += (self.SGDstep * (delt_u - self.lamda * self.u[uid]))
        self.v[iid] += (self.SGDstep * (delt_v - self.lamda * self.v[iid]))
        return self

    def averageEmbedding(self):
        self.u_avg = np.mean(np.array([u for u in self.u.values()]), axis = 0)
        self.v_avg = np.mean(np.array([v for v in self.v.values()]), axis = 0)

    def predict(self, uid, iid, distribution = True):
        """
        predict rating matrix entry given userID and itemID,
        distribution == True when probability distribution is output
        """
        self.initialize(uid, iid, predict = True)   # set avg embeddings for cold-start entries
        m = np.sum(np.multiply(self.u[uid], self.v[iid]), axis=1)
        return softmaxOutput(m, distribution=distribution)

    def modelconfigStore(self, modelconfigurefile = None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        paras = {"u": self.u, "v": self.v, "u_avg": self.u_avg, "v_avg": self.v_avg}
        with open(modelconfigurefile, "w") as f:
            cPickle.dump(paras, f)

    def modelconfigLoad(self, modelconfigurefile = None):
        if modelconfigurefile is None:
            modelconfigurefile = self.modelconfigurefile
        with open(modelconfigurefile, "r") as f:
            paras = cPickle.load(f)
        # write to model parameters #
        self.u = paras["u"]
        self.v = paras["v"]
        self.u_avg = paras["u_avg"]
        self.v_avg = paras["v_avg"]
        self.L = self.u_avg.shape[0]
        print "model successfully loaded from " + modelconfigurefile
