import pandas as pd
import numpy as np
import timeit
# import scipy
from scipy import optimize
from scipy import linalg
from scipy.spatial import distance
import matplotlib.pyplot as plt
import tkinter
# import cos
# import appr
import week_3

if __name__ == '__main__':
    path0 = "/home/stam/workspace/ml/data/20120831_223029_Export_demo_date__prestashop_Cat_All.csv"
    path1 = "/home/stam/workspace/ml/test.txt"
    path2 = "/home/stam/workspace/ml/out.csv"
    path3 = "/home/stam/workspace/ml/data/20120831_223029_Export_demo_date__prestashop_Cat_All.csv"
    # # ret = pd.read_csv(path)
    # # print(ret.head())
    # # print(ret.shape)
    # # print(ret.info())
    # # frame = pd.DataFrame({'numbers':range(10),'chars':['a']*10})
    # # print(frame)
    # frame_ = pd.read_csv(path3,header=0, sep=';')
    # print(frame_)
    # print(frame_.columns)
    # print(frame_.shape)
    #
    # new_line = {'CategoryID': 'F','Active':'1','CategoryName':'9','CategoryParentID':'4','UrlRewrite':'4'}
    # frame_ = frame_.append(new_line,ignore_index=True)
    # print(frame_)
    #
    # # ret.to_csv('out.csv',sep=';',header=True,index=None)
    # # frame_['IsPPPPPPP'] = 1
    # # print(frame_)
    # frame_ = frame_.drop([6], axis=0, inplace=True)
    # # print(frame_)
    # frame_.drop('CategoryID',axis=1, inplace=True)
    # print(frame_)

    # frame_ = pd.read_csv(path3,header=0, sep=';')
    # print(frame_)
    # print (frame_.dtypes)
    # frame_.CategoryID = frame_.CategoryID.apply(pd.to_datetime)
    # print(frame_)
    # frame_.info()
    # frame_.fillna('ddddddd',inplace=True)
    # print(frame_)
    # print(frame_.UrlRewrite)
    # ret1 = frame_[['CategoryID','UrlRewrite']]
    # ret2 = frame_.head(3)
    # ret3 = frame_[:5]
    # ret4 = frame_[5:]
    # print(ret3)
    # print(ret4)
    # ret5 = frame_.loc[[1,3,5],['CategoryID','UrlRewrite']]
    # print(ret5)
    # ret6 = frame_.ix[[1,3,5],['CategoryID','UrlRewrite']]
    # ret7 = frame_.ix[[1,3,5],[1,2]]
    # print(ret6)
    # print(ret7)
    # ret8 = frame_[frame_.CategoryID >= 1011]
    # print(ret8)
    # ret9 = frame_[(frame_.CategoryName == 'третий уровень вложенности') & (frame_.UrlRewrite =='tretij-uroven-vlozhennosti_p')]
    # ret10 = frame_[(frame_.CategoryName == 'третий уровень вложенности') | (frame_.UrlRewrite =='tretij-uroven-vlozhennosti_p') | (frame_.UrlRewrite =='Tretij-uroven-vlozhennosti')]
    # print(ret9)
    # print(ret10)
    # print(frame_.size)
    # print(frame_.shape)
    # frame_
    # x = [2,3,4,5]
    # x1 = (5,6,7,8)
    # y = np.array(x)
    # y1 = np.array(x1)
    # print(type(x))
    # print(type(y))
    #
    # print(type(x1))
    # print(type(y1))
    # print(y[3:4])
    # print(y[[0,1]])
    # print(y[y>3])
    # print(y ** 5)
    # matrix = [[1,2,3],[4,5,6]]
    # matrix_ = np.array(matrix)
    # print(matrix)
    # print(matrix[0][1])
    # print(matrix_[0,1])
    # print(np.random.rand(5,5))
    # start = timeit.timeit()
    # print(np.arange(0,8,0.1))
    # stop = timeit.timeit()
    # print((stop - start)* 1000000)
    # ret = np.arange(1, 101) ** 2
    # print(ret)

    # def f(x):
    #     return  (x**2 -1)
    # x_min = optimize.minimize(f,[10])
    # print(x_min);
    # print(x_min.x)



    # dt = 0.01
    # t = np.arange(0, 30, dt)
    # nse1 = np.random.randn(len(t))  # white noise 1
    # nse2 = np.random.randn(len(t))  # white noise 2
    # s1 = np.sin(2 * np.pi * 10 * t) + nse1
    # s2 = np.sin(2 * np.pi * 10 * t) + nse2
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(t, s1, t, s2)
    # axs[0].set_xlim(0, 2)
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('s1 and s2')
    # axs[0].grid(True)
    #
    # cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
    # axs[1].set_ylabel('coherence')
    #
    # fig.tight_layout()
    # plt.show()
    #

    # a = np.array([[1,2,3],[4,5,6],[1,5,-7]])
    # b = np.array([7,8,9])
    #
    # x = linalg.solve(a,b)
    # print(x)
    # print(np.dot(a,x))
    #


    # plt.plot([1,2,3,4],[1,2,3,4])
    # # plt.show()
    # dt0 = 0.5
    # dt1 = 0.01
    # x1 = np.arange(0, 10, dt0)
    # y1 = np.sin(x1)
    # from scipy import interpolate
    # interpolate_ = interpolate.interp1d(x1,y1)
    # x2 = np.arange(0, 9.5, 0.1)
    # y2 = interpolate_(x2)
    # plt.plot(x1,y1,'o',x2,y2,'-')
    # plt.show()


    # def y1(x1):
    #     return  x1**2

    # plt.plot(x1,y1)
    # plt.show()

    # help(interpolate.interp1d)

    # a = np.array([1,2,0])
    # b = np.array([2,0,1])
    # n = 3
    # k = 5

    # res = a*n + b*k
    #
    # print(a*n)
    # print(b*k)
    # print(a+b)
    # print(res)
    #
    #
    # b =  a[np.newaxis, :]
    # b = a.reshape((1, 3))
    # print(b)
    # print(c)
    #
    # c = np.linalg.norm(a, ord=2) - np.linalg.norm(b, ord=2)
    # print(c)
    #
    # c  = distance.cdist(b, a, metric='euclidean')
    # c  = distance.cdist(a[:, np.newaxis], b[:, np.newaxis], metric='euclidean')
    # c = a*b
    # print(c)
    # print(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

    # c  = distance.cdist(a[np.newaxis, :], b[np.newaxis, :], metric='euclidean') # true
    # print(np.dot(a,b))
    # print(a.dot(b))
    # b = a.reshape((3, 1))
    # c = a[:, np.newaxis]





























