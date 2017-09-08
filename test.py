"""
test dictionary find
"""
import random
import numpy as np
from datetime import datetime
import sys
from datetime import timedelta
import ast

a = {}
b = np.array([2,3])
c = {"h":1, "l":2}
a["b"] = b
a["c"] = c
print a, b, c
b[0] += 2
print a
a["b"][1] += 2
print b
c["h"] += 2
print a
a["c"]["l"] += 2
print c