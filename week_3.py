import re
import numpy as np
import math as mth
import scipy.optimize
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg

# print(scipy.optimize.show_options())

# x = np.arange(1,30,0.01)
# Dx = range(1,30,1)
# x_ = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0])
# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# def fun(x):

fun = lambda x: (np.sin(x / 5.0) * np.exp(x / 10.0) + 5.0 * np.exp(-(x) / 2.0))
f = lambda x: (x - 2) * (x + 1)**2
#
#
# def fun_1(x):
#     return (np.sin(x[:-1] / 5.0) * np.exp(x[:-1] / 10.0) + 5.0 * np.exp(-(x[:-1]) / 2.0))

# print(Dx)

# def f(x):
#     return fun[Dx.index(x)]

# def rosen(x):
#      return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# print(f(2.0))
res = scipy.optimize.minimize_scalar(f)
print(res)


