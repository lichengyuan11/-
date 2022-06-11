# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:13:01 2022

@author: 李程远
"""

import numpy as np
from case33bw_mobi import case33bw
from pypower.api import runpf, ppoption
import csv
import matplotlib.pyplot as plt
import time

def data_test(sample, data_min, data_max, imax):
    t = sample.shape[1]
    sample = np.concatenate([sample, sample[-32:]], axis=0)
    x = np.zeros([166, t], dtype='float32')
    d_index = [0, 33, 67, 68, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 
           86, 87, 88, 91, 92, 93, 98, 100, 133]
    k = 0
    for i in range(166):
        if not (i in d_index):
            x[i] = sample[k]
            k = k + 1
        x[i] = (x[i]*(data_max[i] - data_min[i]) + data_max[i] + data_min[i])/2
    
    bus_load = [i for i in range(2, 34)]

    bus_solar = [7, 8, 24, 25, 30, 32]
    bus_wind = [4, 14, 29, 31]

    mpc = case33bw()
    nbus = np.size(mpc["bus"], 0)
    nbranch = np.size(mpc["branch"], 0)

    bus_map = {}
    bus_map["pd"] = 3
    bus_map["qd"] = 4 
    bus_map["vm"] = 8 
    bus_map["va"] = 9 

    gen_map = {}
    gen_map["bus"] = 1
    gen_map["pg"] = 2
    gen_map["qg"] = 3 

    bran_map = {}
    bran_map["R"] = 3
    bran_map["plineF"] = 14
    bran_map["qlineF"] = 15
    bran_map["plineT"] = 16
    bran_map["qlineT"] = 17
        
    error = 0
    result = {}
    result["res"] = np.zeros([1, t], dtype='float32')
    result["loss"] = np.zeros([1, t], dtype='float32')
    result["V_avg"] = np.zeros([1, t], dtype='float32')
    result["e"] = np.zeros([1, t], dtype='float32')
    result["I_rate_mean"] = np.zeros([1, t], dtype='float32')
    result["I_rate_max"] = np.zeros([1, t], dtype='float32')
    result["heavy_rate"] = np.zeros([1, t], dtype='float32')
    
    start = time.time()
    
    for it in range(t):
        mpc = case33bw()
        for b in bus_load:
            mpc["bus"][b-1, bus_map["pd"]-1] = x[b + 99, it]
            mpc["bus"][b-1, bus_map["qd"]-1] = x[b + 132, it]
                        
        igen = 1;
        for b in bus_solar:
            mpc["gen"] = np.concatenate((mpc["gen"], mpc["gen"][0:1]))
            mpc["gencost"] = np.concatenate((mpc["gencost"], mpc["gencost"][0:1]))
            mpc["gen"][igen, gen_map["bus"]-1] =  b
            mpc["gen"][igen, gen_map["pg"]-1] = x[b + 65, it]
            igen = igen + 1
        
        for b in bus_wind:
            mpc["gen"] = np.concatenate((mpc["gen"], mpc["gen"][0:1]))
            mpc["gencost"] = np.concatenate((mpc["gencost"], mpc["gencost"][0:1]))
            mpc["gen"][igen, gen_map["bus"]-1] =  b
            mpc["gen"][igen, gen_map["pg"]-1] = x[b + 65, it]
            igen = igen + 1
        
        ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
        pf, J, _ = runpf(mpc, ppopt)
        e = np.linalg.svd(J, compute_uv=0)[-1]
        if pf["success"] == 0:
            error += 1
            print(error)
        else:
            vm = pf["bus"][1:, bus_map["vm"]-1]
            va = pf["bus"][1:, bus_map["va"]-1]
            p = pf["gen"][0, gen_map["pg"]-1:gen_map["pg"]]
            q = pf["gen"][0, gen_map["qg"]-1:gen_map["qg"]]
            
            V_avg = vm.mean()
            
            pd = pf["bus"][:, bus_map["pd"]-1:bus_map["pd"]]
            pg = np.zeros([nbus, 1])
            pg[pf["gen"][:,gen_map["bus"]-1].astype(int)-1] = pf["gen"][:, gen_map["pg"]-1:gen_map["pg"]]
            loss = 1 - pd.sum()/pg.sum()
            
            plineF = pf["branch"][:,bran_map["plineF"]-1:bran_map["plineF"]]
            plineT = pf["branch"][:,bran_map["plineT"]-1:bran_map["plineT"]]
            r = pf["branch"][:,bran_map["R"]-1:bran_map["R"]]
            lineI = np.sqrt((plineF + plineT)/r)
            I_rate = lineI/imax
            I_rate_mean = I_rate.mean()
            I_rate_max = I_rate.max()
            heavy_rate = np.sum(I_rate>0.8)/32
            
            # for i in range(32):
            #     vm[i] = (vm[i]*2 - data_max[i+1] - data_min[i+1])/(data_max[i+1] - data_min[i+1])
            #     va[i] = (va[i]*2 - data_max[i+34] - data_min[i+34])/(data_max[i+34] - data_min[i+34])
            # p = (p*2 - data_max[66] - data_min[66])/(data_max[66] - data_min[66])
            # q = (q*2 - data_max[99] - data_min[99])/(data_max[99] - data_min[99])
            # loss_vm = (sample[0:32, it] - vm)**2
            # loss_va = (sample[32:64, it] - va)**2
            # loss_p = (sample[64, it] - p)**2
            # loss_q = (sample[75, it] - q)**2
            # result["res"][0, it] = loss_vm.mean() + loss_va.mean() + loss_p + loss_q
            result["loss"][0, it] = loss
            result["V_avg"][0, it] = V_avg
            result["e"][0, it] = e
            result["I_rate_mean"][0, it] = I_rate_mean
            result["I_rate_max"][0, it] = I_rate_max
            result["heavy_rate"][0, it] = heavy_rate
            
        
        if it % 100 == 0:    
            elapsed = (time.time() - start)
            print("{:.2%}".format(it/t), 
                  "Time:{}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed*(t/(it + 1) - 1)))))
        
    error_rate = error/t
    
    return error_rate, result

day_num = 750
# with open('data/4800.csv', 'r', newline='') as csvfile:
# with open('good result1/7800.csv', 'r', newline='') as csvfile:
with open('data/gen_sample.csv', 'r', newline='') as csvfile:
    reader=csv.reader(csvfile)
    i = 0
    data = np.zeros([108, 288*day_num], dtype='float32')
    for row in reader:
        data[i] = row
        i += 1

i = 0
plt.plot(data[66, 288*i:288*(i + 1)], label='PV1')
plt.plot(data[69, 288*i:288*(i + 1)], label='PV2')
plt.legend()
plt.show()
plt.plot(data[65, 288*i:288*(i + 1)], label='WT1')
plt.plot(data[71, 288*i:288*(i + 1)], label='WT2')
plt.legend()
plt.show()
plt.plot(data[76, 288*i:288*(i + 1)], label='LOAD1')
plt.plot(data[107, 288*i:288*(i + 1)], label='LOAD2')
plt.legend()
plt.show()

        
with open('data/data_scale.csv', 'r', newline='') as csvfile:
    reader=csv.reader(csvfile)
    i = 0
    data_scale = np.zeros([172, 2], dtype='float32')
    for row in reader:
        data_scale[i] = row
        i += 1

with open('data/Imax.csv', 'r', newline='') as csvfile:
    reader=csv.reader(csvfile)
    i = 0
    imax = np.zeros([32, 1], dtype='float32')
    for row in reader:
        imax[i] = row
        i += 1

data_max = data_scale[:, :1]
data_min = data_scale[:, 1:]

error_rate, result = data_test(data, data_min, data_max, imax)

with open('data/result.csv', 'w', newline='') as csvfile:
    writer=csv.writer(csvfile)
    # writer.writerows(result["res"])
    writer.writerows(result["loss"])
    writer.writerows(result["V_avg"])
    writer.writerows(result["e"])
    writer.writerows(result["I_rate_mean"])
    writer.writerows(result["I_rate_max"])
    writer.writerows(result["heavy_rate"])