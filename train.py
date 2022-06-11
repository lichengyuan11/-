# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:08:09 2022

@author: 李程远
"""

import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math
from makeYbus import Ybus
# from res_model import D_net, G_net, E_net
from model import D_net, G_net, E_net

n_bus = 69#33 69
con_bus = 20#0 20
gen_bus = 13#10 13
base = 10#100 10
n_features = (n_bus - 1)*3 + 2 + gen_bus - con_bus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_z = 288*8
batch_size = 128
n_epochs = 500
D_steps = 1
lr_G = 1e-4
lr_E = lr_G
lr_D = 4*lr_G
pf_cor0 = 5e-3*base**2
recon_cor0 = 1
alpha_G = 0.2
alpha_D = 0.2
sigma = 5

def label_scale(labels, n):
    l = len(labels)
    nlabels = np.zeros([l, n], dtype='float32')
    for i in range(l):
        m = labels[i]
        nlabels[i, m] = 1
    return nlabels

def train(data, labels, n, data_min, data_max):
    a = Ybus()
    g = torch.tensor(a.real, dtype=torch.float32).to(device)
    b = torch.tensor(a.imag, dtype=torch.float32).to(device)
    
    def pfloss(pf_data):
        N, C, L = pf_data.shape
        pf_data = pf_data.permute(1, 0, 2).reshape(C, -1)
        t = N*L
        pf_data = torch.cat([pf_data, pf_data[-(n_bus-1-con_bus):]], axis=0)
        x = torch.zeros([n_bus*5+1, t], dtype=torch.float32).to(device)
        k = 0
        for i in range(5*n_bus+1):
            if not (i in d_index):
                x[i] = pf_data[k]
                k = k + 1
            x[i] = (x[i]*(data_max[i] - data_min[i]) + data_max[i] + data_min[i])/2
        vm = x[:n_bus].T
        vmij = (vm.unsqueeze(dim=-1).expand(t, n_bus, n_bus))*(vm.unsqueeze(dim=-1).permute(0, 2, 1).expand(t, n_bus, n_bus))
        va = x[n_bus:n_bus*2].T*math.pi/180
        vaij = va.unsqueeze(dim=-1).expand(t, n_bus, n_bus) - va.unsqueeze(dim=-1).permute(0, 2, 1).expand(t, n_bus, n_bus)        
        p = x[n_bus*2:n_bus*3].T - x[n_bus*3+1:n_bus*4+1].T
        p = p/base - torch.sum(vmij*(g*torch.cos(vaij) + b*torch.sin(vaij)), dim=-1)
        q = torch.cat([x[n_bus*3:n_bus*3+1] - x[n_bus*4+1:n_bus*4+2], -x[n_bus*4+2:]]).T
        q = q/base + torch.sum(vmij*(b*torch.cos(vaij) - g*torch.sin(vaij)), dim=-1)
        loss = torch.mean(p**2 + q**2, dim=-1)
        return loss.reshape(N, L).mean(dim=-1, keepdim=True)   
    # los = pfloss(torch.tensor(data[:100], dtype=torch.float32).to(device))
    # print(los.mean().detach().cpu().numpy())
    # class EMAmodel(nn.Module):
    #     def __init__(self, model, decay=0.99):
    #         super(EMAmodel, self).__init__()
    #         self.module = model.eval()
    #         self.decay = decay
    #     def update(self, model):
    #         model = model.eval()
    #         with torch.no_grad():
    #             for ema_v, model_v in zip(self.module.state_dict().values(),
    #                                       model.state_dict().values()):
    #                 ema_v.copy_(self.decay*ema_v+(1-self.decay)*model_v)
    
    D = D_net().cuda()
    G = G_net().cuda()
    E = E_net().cuda()
    # EMA_G = G_net().cuda()
    # EMA_E = E_net().cuda()
    for m in D.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=alpha_D)
            # nn.init.orthogonal_(m.weight)
    for m in G.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, a=alpha_G)
            # nn.init.orthogonal_(m.weight)
    for m in E.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, a=alpha_G)
            # nn.init.orthogonal_(m.weight)
    # EMA_G.load_state_dict(G.state_dict())
    # EMA_E.load_state_dict(E.state_dict())
    # EMAG = EMAmodel(EMA_G)
    # EMAE = EMAmodel(EMA_E)
    
    # print(sum(p.numel() for p in D.parameters()))
    # print(sum(p.numel() for p in G.parameters()))
    # print(sum(p.numel() for p in E.parameters()))

    opt_D = torch.optim.RAdam(D.parameters(), lr=lr_D, betas=[0, 0.99])
    opt_G = torch.optim.RAdam(G.parameters(), lr=lr_G, betas=[0, 0.99])
    opt_E = torch.optim.RAdam(E.parameters(), lr=lr_E, betas=[0, 0.99])
    
    P_real=[]
    P_fake=[]
    discrim_loss=[]
    pf = []
    recon = []
    labels = label_scale(labels, n)
    
    def gloop(Xs, Ys):
        opt_G.zero_grad()
        Zs = truncnorm.rvs(-sigma, sigma, size=[batch_size, dim_z]).astype(np.float32) / sigma
        Zs = torch.tensor(Zs, dtype=torch.float32).to(device)
        loss = nn.MSELoss()
        with torch.no_grad():
            image_real_Z = E(Xs, Ys)        
        image_fake = G(Zs, Ys)
        reconloss = loss(Xs, G(image_real_Z, Ys))
        # reconloss = loss(E(image_fake, Ys), Zs)
        # reconloss = loss(Xs, G(image_real_Z, Ys)) + loss(E(image_fake, Ys), Zs)
        # grad_r = torch.autograd.grad(reconloss, G.parameters(), retain_graph=True)[0]
        # grad_r = torch.norm(grad_r, p=2).detach().cpu().numpy()
        # print(grad_r)
        reconloss = reconloss * recon_cor_g
        pf_loss = torch.mean(pfloss(image_fake)) * pf_cor
        loss_G = - torch.mean(D(image_fake, Ys, Zs)) + pf_loss + reconloss
        loss_G.backward()
        opt_G.step()
    
    def eloop(Xs, Ys):
        opt_E.zero_grad()
        Zs = truncnorm.rvs(-sigma, sigma, size=[batch_size, dim_z]).astype(np.float32) / sigma
        Zs = torch.tensor(Zs, dtype=torch.float32).to(device)
        loss = nn.MSELoss()
        with torch.no_grad():
            image_fake = G(Zs, Ys)
        image_real_Z = E(Xs, Ys)
        reconloss = loss(E(image_fake, Ys), Zs)
        # reconloss = loss(Xs, G(image_real_Z, Ys))
        # reconloss =  loss(E(image_fake, Ys), Zs) + loss(Xs, G(image_real_Z, Ys))        
        # grad_r = torch.autograd.grad(reconloss, E.parameters(), retain_graph=True)[0]
        # grad_r = torch.norm(grad_r, p=2).detach().cpu().numpy()
        # print(grad_r)
        reconloss = reconloss * recon_cor_e
        loss_E = torch.mean(D(Xs, Ys, image_real_Z)) + reconloss
        loss_E.backward()
        opt_E.step()
        
    def dloop(Xs, Ys):
        for i in range(D_steps):
            opt_D.zero_grad()
            Zs = truncnorm.rvs(-sigma, sigma, size=[batch_size, dim_z]).astype(np.float32) / sigma
            Zs = torch.tensor(Zs, dtype=torch.float32).to(device)          
            with torch.no_grad():
                image_fake = G(Zs, Ys)
                image_real_Z = E(Xs, Ys)
                pf_loss = torch.mean(pfloss(image_fake))
            p_real = D(Xs, Ys, image_real_Z)
            p_gen = D(image_fake, Ys, Zs)
            relu = nn.ReLU()
            loss_D = torch.mean(relu(1 + p_gen)
                                + relu(1 - p_real)
                                )
            # loss_D = torch.mean(p_gen - p_real)       
            loss_D.backward()
            opt_D.step()
        p_real_val = torch.mean(p_real).cpu().detach().numpy()
        p_gen_val = torch.mean(p_gen).cpu().detach().numpy()
        discrim_loss_val = p_real_val - p_gen_val
        pf_loss_val = pf_loss.cpu().detach().numpy()
        loss = nn.MSELoss()
        with torch.no_grad():
            reconloss = loss(Xs, G(image_real_Z, Ys)) + loss(E(image_fake, Ys), Zs)
            reconloss = reconloss.cpu().detach().numpy()
        if p_real_val > 5: p_real_val = 5
        if p_real_val < -5: p_real_val = -5
        if p_gen_val > 5: p_gen_val = 5
        if p_gen_val < -5: p_gen_val = -5
        if discrim_loss_val > 10: discrim_loss_val = 10
        if discrim_loss_val < -10: discrim_loss_val = -10
        return p_real_val, p_gen_val, discrim_loss_val, pf_loss_val, reconloss
    
    def cal_grad(Xs, Ys):
        Zs = truncnorm.rvs(-sigma, sigma, size=[batch_size, dim_z]).astype(np.float32) / sigma
        Zs = torch.tensor(Zs, dtype=torch.float32).to(device)          
        image_fake = G(Zs, Ys)
        image_real_Z = E(Xs, Ys)
        p_real = torch.mean(D(Xs, Ys, image_real_Z))
        grad_E = torch.autograd.grad(p_real, E.parameters())[0]
        grad_E = torch.norm(grad_E, p=2).detach().cpu().numpy()
        p_gen = torch.mean(D(image_fake, Ys, Zs))
        grad_G = torch.autograd.grad(p_gen, G.parameters())[0]
        grad_G = torch.norm(grad_G, p=2).detach().cpu().numpy()
        return grad_G, grad_E
    
    it_num = math.floor(n_epochs*2555/batch_size)
    
    gg = 0
    ge = 0
    
    D.train()
    G.train()
    E.train()
    train_size = 2555 - 2555 % batch_size
    
    for it in range(it_num):
        start = it*batch_size % train_size
        end = start + batch_size
        if  start == 0:
            samplei = np.random.choice(2555, 2555, replace=False)
            data = data[samplei]
            labels = labels[samplei]        
        Xs = data[start:end]
        Ys = labels[start:end]
        Xs = torch.tensor(Xs, dtype=torch.float32).to(device)
        Ys = torch.tensor(Ys, dtype=torch.float32).to(device)
        torch.cuda.empty_cache()
        p_real_val, p_gen_val, discrim_loss_val, pf_loss_val, reconloss = dloop(Xs, Ys)
        
        grad_G, grad_E = cal_grad(Xs, Ys)
        beta = 0.95
        gg = gg*beta + grad_G*(1 - beta)
        ggd = gg/(1 - beta**(it+1))
        ge = ge*beta + grad_E*(1 - beta)
        ged = ge/(1 - beta**(it+1))
        # print(ggd, ged)
        pf_cor = pf_cor0/(1 + 5e5*ggd**2)
        recon_cor_g = recon_cor0/(1 + 5e5*ggd**2)
        recon_cor_e = recon_cor0/(1 + 1e3*ged**2)
        
        gloop(Xs, Ys)
        eloop(Xs, Ys)
        P_real.append(p_real_val)
        P_fake.append(p_gen_val)
        discrim_loss.append(discrim_loss_val)
        pf.append(pf_loss_val)
        recon.append(reconloss)
        
        print("itration {}/{} end".format(it, it_num))
        print("Average P(real)=", p_real_val)
        print("Average P(gen)=", p_gen_val)
        print("Discrim loss:", discrim_loss_val)
        print("powerflow loss:", pf_loss_val)
        print("recon loss:", reconloss)
        if it % 200 == 0:
            G.eval()
            E.eval()
            PATH = 'model/G%s.pth' %it
            torch.save(G.state_dict(), PATH)
            PATH = 'model/E%s.pth' %it
            torch.save(E.state_dict(), PATH)
            with torch.no_grad(): 
                sample_num = 50
                gen_sample = np.zeros([n_features, 288*n*sample_num], dtype='float32')
                for i in range(n):            
                    Ys = np.ones([sample_num, 1], dtype='int')*i
                    Ys = label_scale(Ys, n)
                    Ys = torch.tensor(Ys, dtype=torch.float32).to(device)
                    Zs = truncnorm.rvs(-sigma, sigma, size=[sample_num, dim_z]).astype(np.float32) / sigma
                    Zs = torch.tensor(Zs, dtype=torch.float32).to(device)
                    gen_sample[:, i*288*sample_num:(i+1)*288*sample_num] = G(Zs, Ys).permute(1, 0, 2).reshape(n_features, -1).detach().cpu().numpy()
            with open('data/%s.csv' %it, 'w', newline='') as csvfile:
                writer=csv.writer(csvfile)
                for i in range(n_features):
                    writer.writerows([gen_sample[i, :]])
        
    fig = plt.figure()
    plt.plot(P_real,label="real")
    plt.plot(P_fake,label="fake")
    plt.xlabel('itration')
    plt.legend()
    plt.show()
    fig.savefig("1.png")
    
    fig = plt.figure()
    plt.plot(discrim_loss,label="discrim_loss")
    plt.xlabel('itration')
    plt.legend()
    plt.show()
    fig.savefig("2.png")
    
    fig = plt.figure()
    plt.plot(pf,label="pf_loss")
    plt.xlabel('itration')
    plt.legend()
    plt.show()
    fig.savefig("3.png")
    
    fig = plt.figure()
    plt.plot(recon,label="recon_loss")
    plt.xlabel('itration')
    plt.legend()
    plt.show()
    fig.savefig("4.png")


day_num = 2555

with open('data/sample_69.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    data = np.zeros([7+n_bus*5, 288*day_num], dtype='float32')
    for row in reader:
        data[i] = row
        i += 1
data_min = data.min(1)
data_max = data.max(1)
for i in range(7+n_bus*5):
    data[i] = (data[i]*2 - data_min[i] - data_max[i])/(data_max[i] - data_min[i] + 1e-8)
d_index = []
for i in range(7+n_bus*5):
    if data_min[i] == data_max[i]:
        d_index.append(i)
data = np.delete(data, d_index, axis=0)
data = data[:n_features].reshape(n_features, day_num, 288)
data = np.transpose(data, (1, 0, 2))

with open('data/labeln2_69.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    labels = np.zeros([2555, 1], dtype='int')
    i = 0
    for row in reader:
        if i == 0:
            labels[:, 0] = row
        i += 1

train(data, labels, 5, data_min, data_max)

import dis_mat
