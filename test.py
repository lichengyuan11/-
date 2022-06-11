
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:05:56 2022

@author: 李程远
"""

import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from res_model import G_net, E_net
# from model import G_net, E_net

n_features = 199
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_z = 69*9

def label_scale(labels, n):
    l = len(labels)
    nlabels = np.zeros([l, n], dtype='float32')
    for i in range(l):
        m = labels[i]
        nlabels[i, m] = 1
    return nlabels

def test(data, labels, n, data_min, data_max):    
    labels = label_scale(labels, n)
    sigma = 5
    G = G_net().cuda()
    E = E_net().cuda()
    PATH = 'model/G7200.pth'
    G.load_state_dict(torch.load(PATH))
    PATH = 'model/E7200.pth'
    E.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        G.eval()
        E.eval()
        X = torch.tensor(data[:1250], dtype=torch.float32).to(device)
        Y = torch.tensor(labels[:1250], dtype=torch.float32).to(device)
        Zbar1 = E(X, Y).detach().cpu().numpy()
        Xbar1 = G(E(X, Y), Y).detach().cpu().numpy()
        X = torch.tensor(data[1250:], dtype=torch.float32).to(device)
        Y = torch.tensor(labels[1250:], dtype=torch.float32).to(device)        
        Zbar2 = E(X, Y).detach().cpu().numpy()
        Xbar2 = G(E(X, Y), Y).detach().cpu().numpy()
        Xbar = np.concatenate([Xbar1, Xbar2], axis=0)
        Zbar = np.concatenate([Zbar1, Zbar2], axis=0)
    with torch.no_grad(): 
        G.eval()
        E.eval()
        sample_num = 150
        gen_sample = np.zeros([n_features, 288*n*sample_num], dtype='float32')
        for i in range(n):
            Ys = np.ones([sample_num, 1], dtype='int')*i
            Ys = label_scale(Ys, n)
            Ys = torch.tensor(Ys, dtype=torch.float32).to(device)
            Zs = truncnorm.rvs(-sigma, sigma, size=[sample_num, dim_z]).astype(np.float32) / sigma
            Zs = torch.tensor(Zs, dtype=torch.float32).to(device)
            gen_sample[:, i*288*sample_num:(i+1)*288*sample_num] = G(Zs, Ys).permute(1, 0, 2).reshape(n_features, -1).detach().cpu().numpy()
    with open('data/gen_sample.csv', 'w', newline='') as csvfile:
        writer=csv.writer(csvfile)
        for i in range(n_features):
            writer.writerows([gen_sample[i, :]])
    return Xbar, Zbar

day_num = 2555

with open('data/sample_69.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    data = np.zeros([352, 288*day_num], dtype='float32')
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
data = data[:n_features].reshape(n_features, day_num, 288)
data = np.transpose(data, (1, 0, 2))

with open('data/labeln2_69.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    labels = np.zeros([day_num, 1], dtype='int')
    i = 0
    for row in reader:
        if i == 0:
            labels[:, 0] = row
        i += 1

# labels = np.zeros([2555, 1], dtype='int')

Xbar, Zbar = test(data, labels, 5, data_min, data_max)
plt.hist(Zbar.reshape(-1), 500, [-1, 1], histtype='step', label='Zbar')
Zs = truncnorm.rvs(-5, 5, size=2555*dim_z).astype(np.float32) / 5
plt.hist(Zs, 500, [-1, 1], histtype='step', label='Z')
plt.legend()
plt.show()
# plt.scatter(Xbar, data)
# plt.show()
# i = 0
# plt.plot(data[i, 65], label='WT1')
# plt.plot(data[i, 71], label='WT2')
# plt.legend()
# plt.show()
# plt.plot(Xbar[i, 65], label='WT1')
# plt.plot(Xbar[i, 71], label='WT2')
# plt.legend()
# plt.show()

i = 0
plt.plot(data[i, 137], label='WT1')
plt.plot(data[i, 140], label='WT2')
plt.legend()
plt.show()
plt.plot(Xbar[i, 137], label='WT1')
plt.plot(Xbar[i, 140], label='WT2')
plt.legend()
plt.show()