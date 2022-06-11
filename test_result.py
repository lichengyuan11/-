# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:24:44 2022

@author: 李程远
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_data():
    day_num = 2555
    with open('data/sample.csv', 'r', newline='') as csvfile:
        reader=csv.reader(csvfile)
        i = 0
        data = np.zeros([172, 288*day_num])
        for row in reader:
            data[i] = row
            i += 1
    x_min = data.min(1)
    x_max = data.max(1)
    return data[-6:], x_min[-6:], x_max[-6:]
    
data, data_min, data_max = get_data()
f = data
f_scaled = np.zeros(f.shape)
for i in range(6):
    f_scaled[i] = (f[i]*2 - data_min[i] - data_max[i])/(data_max[i] - data_min[i])
    
f_scaled = f_scaled.reshape(6, -1, 288).transpose(1, 0, 2)

f_t = np.zeros([2555, 6])

for i in range(2555):
    f_t[i, 0] = f_scaled[i, 0].mean()
    f_t[i, 1] = f_scaled[i, 1].mean()
    f_t[i, 2] = f_scaled[i, 2].min()
    f_t[i, 3] = f_scaled[i, 3].mean()
    f_t[i, 4] = f_scaled[i, 4].max()
    f_t[i, 5] = f_scaled[i, 5].mean()

clf = PCA(2)
clf.fit(f_t[:, :5])

sample_num = 150
n = 5

with open('data/result.csv', 'r', newline='') as csvfile:
    reader=csv.reader(csvfile)
    i = 0
    result = np.zeros([6, 288*sample_num*n], dtype='float32')
    for row in reader:
        result[i] = row
        i += 1
        
# f = result[1:]
f_scaled = np.zeros(f.shape)
        
for i in range(6):
    f_scaled[i] = (f[i]*2 - data_min[i] - data_max[i])/(data_max[i] - data_min[i])

f_scaled = f_scaled.reshape(6, -1, 288).transpose(1, 0, 2)

f_t = np.zeros([sample_num*n, 6])

for i in range(sample_num*n):
    f_t[i, 0] = f_scaled[i, 0].mean()
    f_t[i, 1] = f_scaled[i, 1].mean()
    f_t[i, 2] = f_scaled[i, 2].min()
    f_t[i, 3] = f_scaled[i, 3].mean()
    f_t[i, 4] = f_scaled[i, 4].max()
    f_t[i, 5] = f_scaled[i, 5].mean()
    
f_e = clf.transform(f_t[:, :5])

# plt.scatter(f_e[:sample_num, 0], f_e[:sample_num, 1], c='b', label='heavy rate>0')
# plt.scatter(f_e[sample_num:, 0], f_e[sample_num:, 1], c='r', label='heavy rate=0')
plt.scatter(f_e[:sample_num, 0], f_e[:sample_num, 1], c='r', label='c1')
plt.scatter(f_e[sample_num:2*sample_num, 0], f_e[sample_num:2*sample_num, 1], c='b', label='c2')
plt.scatter(f_e[2*sample_num:3*sample_num, 0], f_e[2*sample_num:3*sample_num, 1], c='g', label='c3')
plt.scatter(f_e[3*sample_num:4*sample_num, 0], f_e[3*sample_num:4*sample_num, 1], c='c', label='c4')
plt.scatter(f_e[4*sample_num:, 0], f_e[4*sample_num:, 1], c='k', label='c5')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

i1 = (f_t[:, 1] < -0.05)
i2 = (f_t[:, 1] < 0.06)^(f_t[:, 1] < -0.05)
i3 = (f_t[:, 1] < 0.14)^(f_t[:, 1] < 0.06)
i4 = (f_t[:, 1] < 0.22)^(f_t[:, 1] < 0.14)
i5 = (f_t[:, 1] >= 0.22)
imat = np.zeros([5, 5])
for i in range(5):
    imat[0, i] = i1[sample_num*i:sample_num*(i+1)].sum()
    imat[1, i] = i2[sample_num*i:sample_num*(i+1)].sum()
    imat[2, i] = i3[sample_num*i:sample_num*(i+1)].sum()
    imat[3, i] = i4[sample_num*i:sample_num*(i+1)].sum()
    imat[4, i] = i5[sample_num*i:sample_num*(i+1)].sum()