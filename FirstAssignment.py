import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
#%matplotlib inline

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        total_cost += (f_wb - y[i])**2
    total_cost /= (2 * m)

    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_db += f_wb - y[i]
        dj_dw += (f_wb - y[i]) * x[i]
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db