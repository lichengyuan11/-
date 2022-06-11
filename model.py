# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:39:07 2022

@author: 李程远
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm as sn


n_bus = 69#33 69
con_bus = 20#0 20
gen_bus = 13#10 13
base = 10#100 10
n_features = (n_bus - 1)*3 + 2 + gen_bus - con_bus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 5
kernel_size = 5
padding = kernel_size//2
pm = 'zeros'
dim_z = 288*8
size1 = 1024
size2 = size1
size3 = size2
size4 = int(size3/np.sqrt(2))
size5 = int(size4/np.sqrt(2))
size6 = int(size5/np.sqrt(2))
size7 = int(size6/np.sqrt(2))
size8 = int(size7/np.sqrt(2))
size9 = size8
size10 = size9

alpha_G = 0.2
alpha_D = 0.2

class selfatt(nn.Module):
    def __init__(self, in_channels):
        super(selfatt, self).__init__()
        self.in_channels = in_channels
        self.q = nn.Conv1d(in_channels+1, (in_channels+1)//8, 1, 1, bias=False)
        self.k = nn.Conv1d(in_channels+1, (in_channels+1)//8, 1, 1, bias=False)
        self.v = nn.Conv1d(in_channels+1, in_channels, 1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x0):
        l = x0.size(-1)
        w = torch.tensor([i for i in range(l)]).to(device)
        w = w/l*2 - 1
        w = w.unsqueeze(0).unsqueeze(0).expand(x0.size(0), 1, l)
        x = torch.cat([x0, w], dim=1)
        q = self.q(x).permute(0, 2, 1)
        k = self.k(x)
        v = self.v(x)
        att = torch.bmm(q, k)
        att = self.softmax(att)
        out = torch.bmm(v, att.permute(0, 2, 1))
        return self.gamma * out + x0

class resblock_D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_D, self).__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(alpha_D),
            sn(nn.Conv1d(in_channels, out_channels, kernel_size+1, 2, padding, padding_mode=pm)),
            )
    def forward(self, x):
        return self.mlp(x)

class resblock_E(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_E, self).__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(in_channels+n, out_channels, kernel_size+1, 2, padding, padding_mode=pm),
            )
    def forward(self, x, y):
        y = y.unsqueeze(dim=-1)
        y = y.expand([x.size(0), n, x.size(-1)])
        return  self.mlp(torch.cat([x, y], dim=1))
    
class resblock_G(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_G, self).__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.ConvTranspose1d(in_channels+n, out_channels, kernel_size+1, 2, padding, padding_mode=pm),
            )
        
    def forward(self, x, y):
        y = y.unsqueeze(dim=-1)
        y = y.expand([x.size(0), n, x.size(-1)])
        x = self.mlp(torch.cat([x, y], dim=1))
        return x
           
class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.conv0 = nn.Sequential(
            sn(nn.Conv1d(n_features, size10, 1)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Conv1d(size10, size9, kernel_size, 1, padding, padding_mode=pm)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Conv1d(size9, size8, kernel_size, 1, padding, padding_mode=pm)),
            )
        self.res1 = resblock_D(size8, size7)
        self.att1 = selfatt(size7)
        self.res2 = resblock_D(size7, size6)
        self.att2 = selfatt(size6)
        self.res3 = resblock_D(size6, size5)
        self.res4 = resblock_D(size5, size4)
        self.res5 = resblock_D(size4, size3)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(alpha_D),
            sn(nn.Conv1d(size3, size2, kernel_size, 1, padding, padding_mode=pm)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Conv1d(size2, size1, kernel_size, 1, padding, padding_mode=pm)),
            )
        self.mlp = nn.Sequential(
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(size1*9, dim_z)),
            nn.LeakyReLU(alpha_D),
            )
        self.p1 = nn.Sequential(
            sn(nn.Linear(dim_z, dim_z*2)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z*2, dim_z*2)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z*2, dim_z)),
            nn.LeakyReLU(alpha_D)
            )
        self.p = nn.Sequential(
            sn(nn.Linear(dim_z*2, dim_z*4)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z*4, dim_z*4)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z*4, dim_z*2)),
            nn.LeakyReLU(alpha_D),
            )
        self.yp = nn.Sequential(
            sn(nn.Linear(dim_z*2, dim_z*2)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z*2, n))
            )
        self.yx = nn.Sequential(
            sn(nn.Linear(dim_z, dim_z)),
            nn.LeakyReLU(alpha_D),
            sn(nn.Linear(dim_z, n))
            )
        self.xt = sn(nn.Linear(dim_z, 1))
        self.p1t = sn(nn.Linear(dim_z, 1))
        self.pt = sn(nn.Linear(dim_z*2, 1))

    def forward(self, Xs, Ys, Zs):
        X = self.conv0(Xs)
        X = self.res1(X)
        X = self.att1(X)
        X = self.res2(X)
        X = self.att2(X)
        X = self.res3(X)
        X = self.res4(X)
        X = self.res5(X)
        X = self.conv1(X).reshape(Xs.size(0), -1)
        X = self.mlp(X)
        p1 = self.p1(Zs)
        p = self.p(torch.cat([X, p1], dim=1))
        yp = Ys.unsqueeze(dim=1).bmm(self.yp(p).unsqueeze(dim=-1))
        yp = yp.squeeze(dim=-1)
        yx = Ys.unsqueeze(dim=1).bmm(self.yx(X).unsqueeze(dim=-1))
        yx = yx.squeeze(dim=-1)
        return self.xt(X) + self.p1t(p1) + self.pt(p) + yp + yx

class E_net(nn.Module):
    def __init__(self):
        super(E_net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(n_features+n, size10, 1),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size10, size9, kernel_size, 1, padding, padding_mode=pm),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size9, size8, kernel_size, 1, padding, padding_mode=pm),
            )
        self.res1 = resblock_E(size8, size7)
        self.att1 = selfatt(size7)
        self.res2 = resblock_E(size7, size6)
        self.att2 = selfatt(size6)
        self.res3 = resblock_E(size6, size5)
        self.res4 = resblock_E(size5, size4)
        self.res5 = resblock_E(size4, size3)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size3, size2, kernel_size, 1, padding, padding_mode=pm),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size2, size1, kernel_size, 1, padding, padding_mode=pm),
            )
        self.mlp = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.Linear(size1*9+n, dim_z),
            nn.Tanh()
            )

    def forward(self, Xs, Ys):
        y = Ys.unsqueeze(dim=-1)
        y = y.expand([Xs.size(0), n, 288])
        X = self.conv0(torch.cat([Xs, y], dim=1))
        # X = self.conv0(Xs)
        X = self.res1(X, Ys)
        X = self.att1(X)
        X = self.res2(X, Ys)
        X = self.att2(X)
        X = self.res3(X, Ys)
        X = self.res4(X, Ys)
        X = self.res5(X, Ys)
        X = self.conv1(X).reshape(Xs.size(0), -1)
        X = self.mlp(torch.cat([X, Ys], dim=1))
        return X


class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_z+n, size1*9)
            )
        self.conv0 = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size1, size2, kernel_size, 1, padding, padding_mode=pm),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size2, size3, kernel_size, 1, padding, padding_mode=pm),
            )
        self.res1 = resblock_G(size3, size4)
        self.res2 = resblock_G(size4, size5)
        self.res3 = resblock_G(size5, size6)
        self.att1 = selfatt(size6)
        self.res4 = resblock_G(size6, size7)
        self.att2 = selfatt(size7)
        self.res5 = resblock_G(size7, size8)
        self.conv1 = nn.Sequential(
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size8+n, size9, kernel_size, 1, padding, padding_mode=pm),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size9, size10, kernel_size, 1, padding, padding_mode=pm),
            nn.LeakyReLU(alpha_G),
            nn.Conv1d(size10, n_features, 1),
            nn.Tanh()
            )

    def forward(self, Zs, Ys):
        Zs = self.mlp(torch.cat([Zs, Ys], dim=1)).reshape(Zs.size(0), -1, 9)
        Zs = self.conv0(Zs)
        Zs = self.res1(Zs, Ys)
        Zs = self.res2(Zs, Ys)
        Zs = self.res3(Zs, Ys)
        Zs = self.att1(Zs)
        Zs = self.res4(Zs, Ys)
        Zs = self.att2(Zs)
        Zs = self.res5(Zs, Ys)
        y1 = Ys.unsqueeze(dim=-1)
        y1 = y1.expand([Zs.size(0), n, 288])
        X = self.conv1(torch.cat([Zs, y1], dim=1))
        # X = self.conv1(Zs)
        return X