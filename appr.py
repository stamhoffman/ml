import re
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg


x = np.arange(1,15,0.01)
y = np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)
# plt.plot(x,y)

def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)

# x1 = 1
# x2 = 15
#
# a1 = f(x1)
# a2 = f(x2)
#
# matrix = np.zeros((2,2))
# vector = np.zeros((1,2))
#
# matrix = [[1,1],[1,15]]
# vector = [a1,a2]
#
# print(matrix)
# print(vector)
#
# ret = linalg.solve(matrix,vector)
# print(ret)
#
# y1 = ret[0] - ret[1]*x
#
# plt.plot(x,y,'o',y1,x,'-')
# plt.show()


# x1 = 1
# x2 = 8
# x3  = 15
#
# a1 = f(x1)
# a2 = f(x2)
# a3 = f(x3)
#
#
# print(a1)
# print(a2)
# print(a3)
#
# matrix = np.zeros((3,3))
# vector = np.zeros((1,3))
#
# matrix = [[1,1,1],[1,8,64],[1,15,225]]
# vector = [a1,a2,a3]
#
# print(matrix)
#
# ret = linalg.solve(matrix,vector)
# print(ret)
#
# y1 = ret[0] + ret[1]*x + ret[2]*x*x
# plt.plot(x,y,'o',x,y1,'-')
# plt.show()

#
# x1 = 1
# x2 = 4
# x3  = 10
# x4  = 15
#
# a1 = f(x1)
# a2 = f(x2)
# a3 = f(x3)
# a4 = f(x4)
#
# matrix = np.zeros((4,4))
# vector = np.zeros((1,4))
#
# matrix = [[1,1,1,1],[1,x2,x2**2,x2**3],[1,x3,x3**2,x3**3],[1,x4,x4**2,x4**3]]
# vector = [a1,a2,a3,a4]
#
# print(matrix)
#
# ret = linalg.solve(matrix,vector)
# print(ret)
#
# y1 = ret[0] + ret[1]*x + ret[2]*(x**2) + ret[3]*(x**3)
# plt.plot(x,y,'o',x,y1,'-')
# plt.show()
#
# wr_file = open('submission-2.txt','w')
#
# for unit in ret:
#     wr_file.write(np.array2string(unit))
#     wr_file.write(' ')


x1 = 1
x2 = 4
x3  = 7
x4  = 10
x5  = 15

a1 = f(x1)
a2 = f(x2)
a3 = f(x3)
a4 = f(x4)
a5 = f(x5)

matrix = np.zeros((4,4))
vector = np.zeros((1,4))

matrix = [[1,1,1,1,1],[1,x2,x2**2,x2**3,x2**4],[1,x3,x3**2,x3**3,x3**4],[1,x4,x4**2,x4**3,x4**4],[1,x5,x5**2,x5**3,x5**4]]
vector = [a1,a2,a3,a4,a5]

print(matrix)

ret = linalg.solve(matrix,vector)
print(ret)

y1 = ret[0] + ret[1]*x + ret[2]*(x**2) + ret[3]*(x**3) + ret[4]*(x**4)
plt.plot(x,y,'o',x,y1,'-')
plt.show()