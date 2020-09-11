#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:08:14 2020

@author: yisongdong
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#-------------
# Read CSV Data
#-------------
def ReadCSV(path):
    
    data = pd.read_csv(path)
    
    return data.values


#-------------
# Scale Data
#-------------
def MinMaxScaleData(data):
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data


#-------------
# DataLoader
#-------------
class dataset(Dataset):
    
    def __init__(self,data):
        
        self.samples = []
        
        for i in data:

            self.samples.append(torch.Tensor(i.tolist()))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        return self.samples[idx]
    


#-------------
# Create Dataloader
#-------------
def CreateDataLoader(path, 
                     y_pos,
                     scaled = 0,
                     _batch_size = 32, 
                     _shuffle=True):
    
    data = ReadCSV(path)
    
    neg_idx = list(np.where(data[:,y_pos]==1)[0])
    
    if scaled == 1:
        scaled_data = MinMaxScaleData(data)
        neg_data = scaled_data[neg_idx,1:-1]
        d_set = dataset(neg_data)
        dataloader = DataLoader(d_set,
                                batch_size = _batch_size,
                                shuffle = _shuffle)
        return dataloader
    
    
    neg_data = data[neg_idx,1:-1]
    d_set = dataset(neg_data)
    dataloader = DataLoader(d_set,
                            batch_size = _batch_size,
                            shuffle = _shuffle)
    return dataloader
    


#-------------
# Experimental Data
#-------------
def ExprData(num_feature=29, 
             num_sample=500,
             _batch_size=32,
             _shuffle=True):
    
    x = [np.random.choice([0.8,0.9,1]) for i in range(num_feature)]
    expr_X = np.repeat([x],num_sample,axis=0)
    
    exprdataset = dataset(torch.Tensor(expr_X))
    exprdataloader = DataLoader(exprdataset,
                                batch_size = _batch_size,
                                shuffle = _shuffle)
    
    return exprdataloader



#-------------
# Noise Data
#-------------
def NoiseData(num = 500, 
              _seed = 7,
              _batch_size=32,
              _shuffle=False):
    
    np.random.seed(_seed)
    noises = np.random.normal(size = (num,1))

    noiseset = dataset(noises)
    noiseloader = DataLoader(noiseset,
                             batch_size=_batch_size,
                             shuffle=_shuffle)
    
    return noiseloader