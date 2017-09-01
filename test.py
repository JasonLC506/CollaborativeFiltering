"""
test dictionary find
"""
import random
import numpy as np
from datetime import datetime
import sys
from datetime import timedelta
import ast

a = sys.argv[1]
b = ast.literal_eval(a)
print a, b
print type(a), type(b)
print b[0][0][0]
