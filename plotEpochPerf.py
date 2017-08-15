from matplotlib import pyplot as plt
import numpy as np

filename = "results/log_TD_likeli_SGD-2_K15-15-5_DataMF5"

loss_train = []
loss_valid = []

with open(filename, "r") as f:
    for line in f:
        perf = line.rstrip("\n").split(" ")
        print perf
        if "training:" in perf:
            loss_train.append(float(perf[-1]))
        if "valid:" in perf:
            loss_valid.append(float(perf[-1]))
train = np.array(loss_train)
valid = np.array(loss_valid)

plt.plot(np.arange(train.shape[0]), train, label = "training loss")
plt.plot(np.arange(valid.shape[0]), valid, label = "valid loss")
plt.legend()
plt.savefig(filename + ".png")
plt.show()

