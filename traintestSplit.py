import numpy as np

def split(frac_training,frac_valid, datafile, file_train, file_valid, file_test):
    f_train = open(file_train, "w")
    f_valid = open(file_valid, "w")
    f_test = open(file_test, "w")

    ntrain = 0
    nvalid = 0
    ntest = 0

    with open(datafile, "r") as f:
        for line in f:
            setID = np.argmax(np.random.multinomial(1, [frac_training, frac_valid, 1-frac_training-frac_valid], size = 1))
            if setID == 0:
                f_train.write(line)
                ntrain += 1
            elif setID == 1:
                f_valid.write(line)
                nvalid += 1
            elif setID == 2:
                f_test.write(line)
                ntest += 1
            else:
                print "error"

    print ntrain
    print nvalid
    print ntest


if __name__ == "__main__":
    frac_training = 0.7
    frac_valid = 0.2
    datafile = "data/synthetic_N20000_M10000000_L3_K5"
    split(frac_training = frac_training, frac_valid = frac_valid,
          datafile = datafile,
          file_train = datafile + "_" + str(frac_training)+"train",
          file_valid = datafile + "_" + str(frac_valid)+"valid",
          file_test = datafile + "_" + str(1- frac_training - frac_valid)+"test")
