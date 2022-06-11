# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:34:43 2022

@author: 李程远
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:26:38 2022

@author: 李程远
"""
import numpy as np
from case69 import case
import math
from scipy import interpolate
from pypower.api import runpf, ppoption
import csv
import time

def cal_BBM(mpc):
    nbus = np.size(mpc["bus"],0)
    nbranch = np.size(mpc["branch"],0)
    BBM = np.zeros([nbus, nbranch])
    for ib in range(nbranch):
        fb = int(mpc["branch"][ib, 0])
        tb = int(mpc["branch"][ib, 1])
        BBM[fb - 1, ib] = 1
        BBM[tb - 1, ib] = -1    
    return BBM

def cal_modi_adj(mpc):
    branch = mpc["branch"]
    nbus = np.size(mpc["bus"], 0)

    adj = np.zeros([nbus, nbus])

    for ib in range(np.size(branch, 0)):
        i = int(branch[ib,0]) - 1
        j = int(branch[ib,1]) - 1
        adj[i,j] = 1
        adj[j,i] = 1
    modi_adj = adj + np.eye(nbus)
    return modi_adj


def gen_samples():
    dailyload1_base = np.array([
        [0, 0.78],
        [2, 0.71],
        [4, 0.72],
        [6, 0.76],
        [8, 0.78],
        [10, 0.98],
        [12, 0.95],
        [14, 1],
        [16, 0.97],
        [20, 0.8],
        [23, 0.86],
        [24, 0.78]
        ])
    
    dailyload2_base = np.array([
        [0, 0.78],
        [2, 0.71],
        [4, 0.72],
        [6, 0.76],
        [8, 0.78],
        [10, 0.8],
        [12, 0.8],
        [14, 0.7],
        [16, 0.8],
        [20, 0.9],
        [21, 1],
        [23, 0.86],
        [24, 0.78]
        ])
    
    fload1 = interpolate.interp1d(dailyload1_base[:,0], dailyload1_base[:,1], kind = "cubic")
    fload2 = interpolate.interp1d(dailyload2_base[:,0], dailyload2_base[:,1], kind = "cubic")
    

    with open('data/climate_data_69.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        data = np.array(rows, dtype='float32')
    
    day_num = math.floor(data.shape[0]/288)
    # day_num = 31
    t_num = day_num*288
    bus_load2 = [i for i in range(35, 70)]
    bus_load1 = [i for i in range(2, 35)]

    bus_solar = [11, 12, 21, 49, 50, 59, 61, 64]
    bus_wind = [8, 17, 18, 51, 65]
    p_solar = np.zeros(69)
    p_solar[bus_solar] = [120, 120, 120, 350, 350, 120, 1200, 200]
    p_wind = np.zeros(69)
    p_wind[bus_wind] = [80, 80, 80, 60, 60]

    t_step = 5
    t_end = t_num*t_step - t_step
    t = np.linspace(0, t_end, t_num)

    mpc = case()
    nbus = np.size(mpc["bus"], 0)
    nbranch = np.size(mpc["branch"], 0)
    BBM = cal_BBM(mpc)

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


    raw = {}
    raw["vm"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["va"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["pd"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["pg"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["qd"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["qg"] = np.zeros([1, len(t)], dtype='float32')
    raw["lineI"] = np.zeros([nbranch, len(t)], dtype='float32')
    raw["plineF"] = np.zeros([nbranch, len(t)], dtype='float32')
    raw["plineT"] = np.zeros([nbranch, len(t)], dtype='float32')
    raw["t"] = np.zeros([1, len(t)], dtype='float32')
    raw["e"] = np.zeros([1, len(t)], dtype='float32')
    raw["loss"] = np.zeros([1, len(t)], dtype='float32')

    raw["plineF_pi"] = np.zeros([nbus, len(t)], dtype='float32')
    raw["plineT_pi"] = np.zeros([nbus, len(t)], dtype='float32')
    
    start = time.time()
    
    for it in range(len(t)):
        mpc = case()
        
        day = math.floor(it*300/24/3600)
        hour = it*300/3600 - day*24
        load1 = fload1(hour)
        load2 = fload2(hour)
       

        for b in bus_load1:
            mpc["bus"][b-1, bus_map["pd"]-1] = mpc["bus"][b-1, bus_map["pd"]-1]*load1*data[it, b + 11]
            mpc["bus"][b-1, bus_map["qd"]-1] = mpc["bus"][b-1, bus_map["qd"]-1]*load1*data[it, b + 11]
        
        for b in bus_load2:
            mpc["bus"][b-1, bus_map["pd"]-1] = mpc["bus"][b-1, bus_map["pd"]-1]*load2*data[it, b + 11]
            mpc["bus"][b-1, bus_map["qd"]-1] = mpc["bus"][b-1, bus_map["qd"]-1]*load2*data[it, b + 11]
        
        
        igen = 1;
        for b in bus_solar:
            mpc["gen"] = np.concatenate((mpc["gen"], mpc["gen"][0:1]))
            mpc["gencost"] = np.concatenate((mpc["gencost"], mpc["gencost"][0:1]))
            mpc["gen"][igen, gen_map["bus"]-1] =  b
            mpc["gen"][igen, gen_map["pg"]-1] = p_solar[b] * data[it, igen - 1]/1000
            igen = igen + 1
        
        for b in bus_wind:
            mpc["gen"] = np.concatenate((mpc["gen"], mpc["gen"][0:1]))
            mpc["gencost"] = np.concatenate((mpc["gencost"], mpc["gencost"][0:1]))
            mpc["gen"][igen, gen_map["bus"]-1] =  b
            mpc["gen"][igen, gen_map["pg"]-1] = p_wind[b] * data[it, igen - 1]/1000
            igen = igen + 1
        
        ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
        pf, J, _ = runpf(mpc, ppopt)
        e = np.linalg.svd(J, compute_uv=0)[-1]
        error = 1
        if pf["success"] == 0:
            print("did not converge {}".format(error))
            error += 1

        raw["vm"][:,it] = pf["bus"][:, bus_map["vm"]-1]
        raw["va"][:,it] = pf["bus"][:, bus_map["va"]-1]
        pd = pf["bus"][:, bus_map["pd"]-1:bus_map["pd"]]
        qd = pf["bus"][:, bus_map["qd"]-1:bus_map["qd"]]

        pg = np.zeros([nbus, 1])
        pg[pf["gen"][:,gen_map["bus"]-1].astype(int)-1] = pf["gen"][:, gen_map["pg"]-1:gen_map["pg"]]
        qg = np.zeros([nbus, 1])
        qg[pf["gen"][:,gen_map["bus"]-1].astype(int)-1] = pf["gen"][:, gen_map["qg"]-1:gen_map["qg"]]
        loss = 1 - pd.sum()/pg.sum()


        plineF = pf["branch"][:,bran_map["plineF"]-1:bran_map["plineF"]]
        plineT = pf["branch"][:,bran_map["plineT"]-1:bran_map["plineT"]]
        r = pf["branch"][:,bran_map["R"]-1:bran_map["R"]]
        lineI = np.sqrt((plineF + plineT)/r)



        # raw["plineF"][:,it:it+1] = plineF
        # raw["plineT"][:,it:it+1] = plineT
        raw["lineI"][:, it:it+1] = lineI
        raw["pg"][:, it:it+1] = pg
        raw["qg"][0, it] = qg[0]
        raw["pd"][:, it:it+1] = pd
        raw["qd"][:, it:it+1] = qd
        raw["t"][0, it] = it*5
        raw["e"][0, it] = e
        raw["loss"][0, it] = loss 
        if it % 100 == 0:    
            elapsed = (time.time() - start)
            print("{:.2%}".format(it/t_num), 
                  "Time:{}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed*(t_num/(it + 1) - 1)))))
    # raw["plineF_pi"] = np.dot(BBM, raw["plineF"])
    # raw["plineT_pi"] = np.dot(BBM, raw["plineT"])
    
    return raw

mpc = case()
# modi_adj = cal_modi_adj(mpc)
# BBM = cal_BBM(mpc)
raw = gen_samples()
I_rate = np.zeros([68, raw["e"].shape[-1]])
I_rate_mean = np.zeros([1, raw["e"].shape[-1]])
I_rate_max = np.zeros([1, raw["e"].shape[-1]])
heavy_rate = np.zeros([1, raw["e"].shape[-1]])
V_avg = np.zeros([1, raw["e"].shape[-1]])
I_max = raw["lineI"].max(axis=1)
for i in range(68):
    I_rate[i] = raw["lineI"][i]/I_max[i]
I_rate_mean[0] = I_rate.mean(axis=0)
I_rate_max[0] = I_rate.max(axis=0)
heavy_rate[0] = np.sum(I_rate>0.8, axis=0)/68
V_avg[0] = raw["vm"][1:].mean(axis=0)

with open('data/sample_69.csv', 'w', newline='') as csvfile:
    writer=csv.writer(csvfile)
    for i in range(69):
        writer.writerows([raw["vm"][i, :]])
    for i in range(69):
        writer.writerows([raw["va"][i, :]])
    for i in range(69):
        writer.writerows([raw["pg"][i, :]])
    writer.writerows(raw["qg"])
    for i in range(69):
        writer.writerows([raw["pd"][i, :]])
    for i in range(69):
        writer.writerows([raw["qd"][i, :]])
    
    writer.writerows(raw["loss"])
    writer.writerows(V_avg)
    writer.writerows(raw["e"])
    writer.writerows(I_rate_mean)
    writer.writerows(I_rate_max)
    writer.writerows(heavy_rate)