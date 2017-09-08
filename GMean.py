import numpy as np
from scipy.stats.mstats import gmean
import ast

def Gmeans(result):
    L = len(result)
    recalls = [(item[1]+1.0)/(item[0]+2.0) for item in result.values()]
    return gmean(np.array(recalls))

if __name__ == "__main__":
    result = ast.literal_eval("{0: [1545046, 1442157], 1: [128769, 28667], 2: [144670, 82199], 3: [69492, 18664], 4: [102888, 54366], 5: [221512, 148032]}")
    print Gmeans(result)