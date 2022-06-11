# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:24:00 2022

@author: 李程远
"""

import csv
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_data():
    day_num = 2555
    with open('data/sample_69.csv', 'r', newline='') as csvfile:
        reader=csv.reader(csvfile)
        i = 0
        data = np.zeros([352, 288*day_num])
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
f_t0 = clf.fit_transform(f_t[:, :5])
# index = (f_t[:, 5] == -1)
# indexn = (1 - index).astype(bool)
# plt.scatter(f_t0[indexn, 0], f_t0[indexn, 1], c='b', label='heavy rate>0')
# plt.scatter(f_t0[index, 0], f_t0[index, 1], c='r', label='heavy rate=0')
# index1 = (f_t[:, 1] < -0.05)
# index2 = (f_t[:, 1] < 0.06)^(f_t[:, 1] < -0.05)
# index3 = (f_t[:, 1] < 0.14)^(f_t[:, 1] < 0.06)
# index4 = (f_t[:, 1] < 0.22)^(f_t[:, 1] < 0.14)
# index5 = (f_t[:, 1] >= 0.22)
index1 = (f_t[:, 1] < -0.15)
index2 = (f_t[:, 1] < -0.08)^(f_t[:, 1] < -0.15)
index3 = (f_t[:, 1] < -0.01)^(f_t[:, 1] < -0.08)
index4 = (f_t[:, 1] < 0.06)^(f_t[:, 1] < -0.01)
index5 = (f_t[:, 1] >= 0.06)
plt.scatter(f_t0[index1, 0], f_t0[index1, 1], c='r', label='c1')
plt.scatter(f_t0[index2, 0], f_t0[index2, 1], c='b', label='c2')
plt.scatter(f_t0[index3, 0], f_t0[index3, 1], c='g', label='c3')
plt.scatter(f_t0[index4, 0], f_t0[index4, 1], c='c', label='c4')
plt.scatter(f_t0[index5, 0], f_t0[index5, 1], c='k', label='c5')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()

label = np.zeros(2555, dtype=int)
label[index1] = 0
label[index2] = 1
label[index3] = 2
label[index4] = 3
label[index5] = 4

# with open('data/labeln.csv', 'w', newline='') as csvfile:
#     writer=csv.writer(csvfile)
#     writer.writerows([int(index)])

with open('data/labeln2_69.csv', 'w', newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerows([label])
    