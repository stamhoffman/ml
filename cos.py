import re
import numpy as np
from scipy.spatial import distance
import pandas as pd

df = open('sentences.txt','r')
list_string = []
lists = []
dict_world = dict()
index = 0

for str in df:
    list_string.append(str.lower())

for low_string in list_string:
    list_split = re.split('[^a-z]',low_string)
    no_zero = list()
    for world in list_split:
        if len(world) != 0 and world != ',':
            no_zero.append(world)
            ret = dict_world.get(world)
            if ret == None:
                dict_world[world] = index
                index = index + 1
    lists.append(no_zero)


matrix = np.zeros((len(list_string),len(dict_world)))

for row in range(len(list_string)):
    col = 0
    for world in dict_world:
        for str in lists[row]:
            if world == str:
                matrix[row][col] = matrix[row][col] + 1
        col = col + 1


cos_array = []

for i in range(len(list_string)):
    ret = distance.cosine(matrix[0],matrix[i])
    cos_array.append(ret)
    print('Косинусное растояние между предложением с номером 0 и номером',i,'= ',ret)

find_arr = np.array(cos_array)
ret_index = find_arr.argsort()

wr_file = open('submission-1.txt','w')

wr_file.write(np.array2string(ret_index[1]))
wr_file.write(' ')
wr_file.write(np.array2string(ret_index[2]))


