import numpy as np

s = 0.05
def f(x):
    return -s*x

def g(x):
    delta = 1
    return np.diag((x[0]**2 + delta, x[1]**2 + delta))

def Df(x):
    return np.diag((-s, -s))

