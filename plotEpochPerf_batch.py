from matplotlib import pyplot as plt
import numpy as np

filename_origin = "logs/TD_reaction_NYTWaPoWSJ_K10_0.7train"

for hyperpara in [4,6,7,8,9,10]:

    pattern = {"model_hyperparameters": [hyperpara], "SGDstep": 0.01, "SCALE": 0.1}

    filename = filename_origin + str(pattern["model_hyperparameters"]) + "_" + "SGDstep" + str(pattern["SGDstep"]) + "_" + "SCALE" + str(pattern["SCALE"])

    loss_train = []
    loss_valid = []
    title = "hyperparas" + "[" + str(hyperpara) + "]"
    loss_min = 100
    loss_min_loc = 0

    matched = True
    data_collection = False

    with open(filename, "r") as f:
        for line in f:
            perf = line.rstrip("\n").split(" ")
            # print perf

            # if "model_hyperparameters:" in perf:
            #     if data_collection:
            #         break
            #     matched = True
            #     if str(pattern["model_hyperparameters"]) in line and matched:
            #         title += line.rstrip("\n") + "_"
            #     else:
            #         matched = False
            #         title = ""
            # if "SGDstep:" in perf:
            #     if str(pattern["SGDstep"]) in line and matched:
            #         title += line.rstrip("\n") + "_"
            #     else:
            #         matched = False
            #         title = ""
            # if "SCALE:" in perf:
            #     if str(pattern["SCALE"]) in line and matched:
            #         title += line.rstrip("\n")
            #     else:
            #         matched = False
            #         title = ""
            # if not matched:
            #     continue
            # data_collection = True

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

    plt.plot(np.arange(train.shape[0]), train, label = "training loss" + title)
    plt.plot(np.arange(valid.shape[0]), valid, label = "valid loss" + title)
    # plt.title(title)
plt.legend()
# plt.savefig(filename + ".png")
plt.show()

