from matplotlib import pyplot as plt
import numpy as np

filename = "logs/TD_reaction_NYTWaPoWSJ_K10_0.7train"

pattern = {"model_hyperparameters": [10,10,5], "SGDstep": 0.001, "SCALE": 0.1}

loss_train = []
loss_valid = []
title = ""
loss_min = 100
loss_min_loc = 0

matched = True
data_collection = False

with open(filename, "r") as f:
    for line in f:
        perf = line.rstrip("\n").split(" ")
        # print perf

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
            value = float(perf[-1])
            loss_train.append(value)
        if "valid:" in perf:
            value = float(perf[-1])
            loss_valid.append(value)
            if value < loss_min:
                loss_min = value
                loss_min_loc = loss_valid.index(value)
train = np.array(loss_train)
valid = np.array(loss_valid)

print title, "min valid loss", loss_min, "at epoch", loss_min_loc

plt.plot(np.arange(train.shape[0]), train, label = "training loss")
plt.plot(np.arange(valid.shape[0]), valid, label = "valid loss")
plt.title(title)
plt.legend()
# plt.savefig(filename + ".png")
plt.show()

