"""
test dictionary find
"""
import random
import numpy as np
from datetime import datetime
from datetime import timedelta
from MultiMF import MultiMF
from trainingDataGenerator import datagenerator

g = datagenerator("data/reaction_NYTWaPoWSJ_K10_0.2test")
cnt = 0
for samp in g.sample(random = True):
    print samp
    cnt += 1
    if cnt > 99:
        break
