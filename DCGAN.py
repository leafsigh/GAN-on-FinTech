#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:55:43 2020

@author: yisongdong
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#------------
# Weight Initializer
#------------
class Generator(nn.Module):
    
    def __init__(self, input_shape):
        
        super(Generator,self).__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose1d(input_shape,out_channels = 4, kernel_size = 2, bias = False),
            nn.BatchNorm1d(4), 
            nn.LeakyReLU(0.2, True),)
        
    def block(self):
        
        