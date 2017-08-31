"""
test dictionary find
"""
import random
import numpy as np
from datetime import datetime
from datetime import timedelta
from MultiMF import MultiMF

a = np.arange(6).reshape([2,3])
b = np.arange(4)*1.1
c = np.arange(5)*2.0
print a, b, c
prod = np.outer(np.outer(b, a.reshape(6)).reshape(4*6), c)
print prod.reshape([4,2,3,5])
