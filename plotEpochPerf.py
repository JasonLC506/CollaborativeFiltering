from matplotlib import pyplot as plt
import numpy as np

filename = "logs/MultiMF_TDsynthetic_N500_M500_L3_ku15_kv15_kr5_0.7train"

pattern = {"model_hyperparameters": [10], "SGDstep": 0.01, "SCALE": 0.1}

loss_train = []
loss_valid = []
title = ""

matched = True
data_collection = False

with open(filename, "r") as f:
    for line in f:
        perf = line.rstrip("\n").split(" ")
        print perf

        if "model_hyperparameters:" in perf:
            if data_collection:
                break
            matched = True
            if str(pattern["model_hyperparameters"]) in line and matched:
                title += line.rstrip("\n") + "_"
            else:
                matched = False
                title = ""
        if "SGDstep:" in perf:
            if str(pattern["SGDstep"]) in line and matched:
                title += line.rstrip("\n") + "_"
            else:
                matched = False
                title = ""
        if "SCALE:" in perf:
            if str(pattern["SCALE"]) in line and matched:
                title += line.rstrip("\n")
            else:
                matched = False
                title = ""
        if not matched:
            continue
        data_collection = True
        if "training:" in perf:
            loss_train.append(float(perf[-1]))
        if "valid:" in perf:
            loss_valid.append(float(perf[-1]))
train = np.array(loss_train)
valid = np.array(loss_valid)

plt.plot(np.arange(train.shape[0]), train, label = "training loss")
plt.plot(np.arange(valid.shape[0]), valid, label = "valid loss")
plt.title(title)
plt.legend()
# plt.savefig(filename + ".png")
plt.show()

