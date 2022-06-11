# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:50:01 2022

@author: 李程远
"""

from netCDF4 import Dataset
import numpy as np
import csv
from scipy import interpolate

data = {'temp':np.zeros([0, 35], dtype='float32'), 
        'wind':np.zeros([0, 35], dtype='float32'), 
        'srad':np.zeros([0, 35], dtype='float32')}
for key in ['temp', 'wind', 'srad']:
    for i in range(2, 9):
        for j in range(1, 10):
            nc = Dataset('D:/climate_data/{}/{}_ITPCAS-CMFD_V0106_B-01_03hr_010deg_201{}0{}.nc'
                         .format(key, key, i, j))
            num = nc.variables[key][:].data[:, 243:248, 458:465]
            data[key] = np.concatenate([data[key], num.reshape(num.shape[0], -1)])
        for j in range(0, 3):
            nc = Dataset('D:/climate_data/{}/{}_ITPCAS-CMFD_V0106_B-01_03hr_010deg_201{}1{}.nc'
                         .format(key, key, i, j))
            num = nc.variables[key][:].data[:, 243:248, 458:465]
            data[key] = np.concatenate([data[key], num.reshape(num.shape[0], -1)])


data_num = data['temp'].shape[0]
t_num = (data_num - 1)*36 + 1
t_end = (t_num - 1)*5
t = np.linspace(0, t_end, t_num)
data_t_end = (data_num - 1)*180
data_t = np.linspace(0, data_t_end, data_num)
raw = np.zeros([t_num, 42], dtype='float32')
##solar
solar_b = [7, 8, 24, 25, 30, 32]
for i in range(6):
    srad = data['srad'][:, solar_b[i]]
    f = interpolate.interp1d(data_t, srad, kind = "cubic")
    srad_int = f(t)
    for j in range(len(srad_int)):
        if srad_int[j] >= 0 and srad_int[j] < 120:
            raw[j, i] = srad_int[j]**2/120/800 
        elif srad_int[j] >= 120 and srad_int[j] < 800:
            raw[j, i] = srad_int[j]/800
        elif srad_int[j] >= 800:
            raw[j, i] = 1
##wind
wind_b = [4, 14, 29, 31]
for i in range(4):
    wind = data['wind'][:, wind_b[i]]
    f = interpolate.interp1d(data_t, wind, kind = "cubic")
    wind_int = f(t)
    for j in range(len(wind_int)):
        if wind_int[j] >= 0 and wind_int[j] < 1:
            raw[j, i + 6] = 0 
        elif wind_int[j] >= 1 and wind_int[j] < 5:
            raw[j, i + 6] = wind_int[j]**2*0.0273 + wind_int[j]*0.0861 - 0.1134
        elif wind_int[j] >= 5 and wind_int[j] < 10:
            raw[j, i + 6] = 1
        elif wind_int[j] >= 10:
            raw[j, i + 6] = 0
##load
load_b = range(32)
for i in range(32):
    load = data['temp'][:, load_b[i]]
    f = interpolate.interp1d(data_t, load, kind = "cubic")
    load_int = f(t) - 273.15
    for j in range(len(load_int) - 288):
        load_mean = np.mean(load_int[j:j + 288])
        raw[j + 288, i + 10] = load_mean**2*4.45e-4 - load_mean*6.24e-3 + 0.85
        raw[j + 288, i + 10] = raw[j + 288, i + 10]*(1 + 0.05*np.random.randn(1))
raw = raw[288 + 192:]


with open('data/climate_data.csv', 'w', newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerows(raw)

        

