"""
assume data file will be too big to be read in to memory.
Data is stored line by line with each line a transaction. In most cases, each line is of same length
"""
import numpy as np
import ast

Notation = {"userID": "POSTID", "itemID": "READERID", "labelID": "EMOTICON"} # transforming from posts notation to CF convention

class datagenerator(object):
    def __init__(self, datafile):
        # meta data #
        self.Nuser = 0
        self.Nitem = 0
        self.Nlabel = 0
        self.Ntran = 0
        self.datafile = datafile

        # for random access #
        self.filesize = 0

        # create dictionary to transform simbol data into index #
        self.user_dict = {}
        self.item_dict = {}
        self.label_dict = {}

        with open(self.datafile, "r") as f:
            for line in f.readlines():
                try:
                    transaction = ast.literal_eval(line.rstrip()) # into transaction dictionary
                except:
                    print line
                # extract raw info #
                userID = transaction[Notation["userID"]]
                itemID = transaction[Notation["itemID"]]
                labelID = transaction[Notation["labelID"]]
                # assign index to IDs #
                if userID not in self.user_dict:
                    self.user_dict[userID] = self.Nuser
                    self.Nuser += 1
                if itemID not in self.item_dict:
                    self.item_dict[itemID] = self.Nitem
                    self.Nitem += 1
                if labelID not in self.label_dict:
                    self.label_dict[labelID] = self.Nlabel
                    self.Nlabel += 1
                self.Ntran += 1
                # count filesize #
                self.filesize += len(line)

    def sample(self, replacement = True):
        """
        sampling data with replacement, the case without replacement is not supported currently
        """
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
            # transform into formal input #
            try:
                transaction = ast.literal_eval(random_line.rstrip())
            except:
                print "error"
                print len(random_line)
                print random_line
                break
            userID = transaction[Notation["userID"]]
            itemID = transaction[Notation["itemID"]]
            labelID = transaction[Notation["labelID"]]
            samp = [self.user_dict[userID], self.item_dict[itemID], self.label_dict[labelID]]
            yield samp

    def dictshow(self):
        print "user_dict"
        print self.user_dict
        print "item_dict"
        print self.item_dict
        print "label_dict"
        print self.label_dict

    def N(self):
        return self.Nuser

    def M(self):
        return self.Nitem

    def L(self):
        return self.Nlabel


if __name__ == "__main__":
    training = datagenerator("data/test")
    print "Nuser: ", training.N()
    print "Nitem: ", training.M()
    print "Nlabel: ", training.L()
    for samp in training.sample():
        print samp

    training.dictshow()