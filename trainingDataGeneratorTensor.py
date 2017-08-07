"""
generate tensor data
"""

import numpy as np
import ast

label_dict = {0:0, 1:1, 2:2}

class datageneratorTensor(object):
    def __init__(self, datafile):
        self.datafile = datafile
        self.Nlabel = len(label_dict)
        self.Ntran = 0
        self.label_dict = label_dict

        self.filesize = 0

        with open(self.datafile, "r") as f:
            for line in f:
                self.filesize += len(line)
                self.Ntran += 1

    def sample(self, replacement = True, random = True):
        """
        sampling with replacement, the case without is not supported currently
        """
        if random is False:
            with open(self.datafile, "r") as f:
                for line in f:
                    try:
                        transaction = ast.literal_eval(line.rstrip())
                    except:
                        print line
                        break
                    yield transaction
        else:
            for isamp in xrange(0, self.Ntran):
                offset = np.random.randint(0, self.filesize)
                with open(self.datafile, "r") as f:
                    f.seek(offset)
                    f.readline()
                    random_line = f.readline()
                    # for end line #
                    if len(random_line) == 0:
                        f.seek(0)
                        random_line = f.readline()
                    transaction = ast.literal_eval(random_line.rstrip())
                    yield transaction

    def L(self):
        return self.Nlabel
