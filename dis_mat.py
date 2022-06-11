# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:28:20 2022

@author: 李程远
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

n_features = 199
day_num1 = 2555

with open('data/sample_69.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    data = np.zeros([352, 288*day_num1], dtype='float32')
    for row in reader:
        data[i] = row
        i += 1
data_min = data.min(1)
data_max = data.max(1)
for i in range(352):
    data[i] = (data[i]*2 - data_min[i] - data_max[i])/(data_max[i] - data_min[i] + 1e-8)
d_index = []
for i in range(352):
    if data_min[i] == data_max[i]:
        d_index.append(i)
data = np.delete(data, d_index, axis=0)
data = data[:n_features].reshape(n_features, day_num1, 288)
data = np.transpose(data, (1, 0, 2))

day_num2 = 250

like = []
div = []

for k in range(49):
    with open('data/%s.csv' %(200*k+200), 'r', newline='') as csvfile:
    # with open('data/gen_sample.csv', 'r', newline='') as csvfile:
        reader=csv.reader(csvfile)
        i = 0
        data_gen = np.zeros([n_features, 288*day_num2], dtype='float32')
        for row in reader:
            data_gen[i] = row
            i += 1
    data_gen = data_gen.reshape(n_features, day_num2, 288)
    data_gen = np.transpose(data_gen, (1, 0, 2))
    
    dis_mat1 = np.zeros([day_num2, day_num1])
    
    for i in range(day_num2):
        for j in range(day_num1):
            dis_mat1[i, j] = np.sqrt(np.sum((data_gen[i] - data[j])**2))
            
    dis_mat2 = np.zeros([day_num2, day_num2])
    
    for i in range(day_num2):
        for j in range(i, day_num2):
            dis_mat2[i, j] = np.sqrt(np.sum((data_gen[i] - data_gen[j])**2))
            
    like.append(dis_mat1.min(1).mean())
    div.append(dis_mat2.sum()*2/day_num2/(day_num2 - 1))
    print('{} finished'.format(k))

x = [i*200+200 for i in range(len(like))]
plt.plot(x, like)
plt.show()
plt.plot(x, div)
plt.show()

fig = plt.figure()
plt.plot(x, like, label="D1")
plt.plot(x, div, label="D2")
plt.xlabel('itration')
plt.legend()
plt.show()
fig.savefig("D.png")

        