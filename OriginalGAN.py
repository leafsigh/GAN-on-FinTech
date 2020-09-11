#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:03:31 2020

@author: yisongdong
"""

import torch.nn as nn

#------------
# Discriminator
#------------ 
class Discriminator(nn.Module):
    
    def __init__(self, input_shape:int):
        super(Discriminator,self).__init__()
        
        self.model = nn.Sequential(
            
            nn.Linear(int(input_shape), 64),
            nn.BatchNorm1d(num_features = 64),
            nn.LeakyReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features = 32),
            nn.LeakyReLU(),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(num_features = 16),
            nn.LeakyReLU(),
            
            nn.Linear(16,1),
            nn.Sigmoid())
        
    def forward(self,x):
        
        dis = self.model(x)
        return dis
    
   
#------------
# Generator
#------------ 
class Generator(nn.Module):

    def __init__(self, input_shape:int):
        
        super(Generator,self).__init__()
        
        self.model = nn.Sequential(*self.block(input_shape, 4),
                                   *self.block(4, 8),
                                   *self.block(8, 16),
                                   *self.block(16, 29))
    
    def block(self, input_feat, output_feat, normalize=True):
        
        layer_block = [nn.Linear(input_feat, output_feat)]
        
        if normalize:
            layer_block.append(nn.BatchNorm1d(output_feat))
        layer_block.append(nn.LeakyReLU(0.2, inplace=True))
        
        return layer_block
            
    def forward(self, noise):
        
        gen = self.model(noise)
        return gen