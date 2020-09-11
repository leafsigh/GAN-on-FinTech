#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:10:53 2020

@author: yisongdong
"""

import torch
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
from torch.autograd import Variable
#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import numpy as np



#--------------
# Define Optimizer
#--------------
def Optimizer(g, d, _lr=0.1):
    
    optimizer_G = optim.Adam(g.parameters(), lr=_lr)
    optimizer_D = optim.Adam(d.parameters(), lr=_lr)
    
    return optimizer_G, optimizer_D
    

#--------------
# Generate Fake Data
#--------------
def GenerateFake(noiseloader, g):
    
    for i,batch in enumerate(noiseloader):
        if i==0:
            fakes = g(batch[:,0].reshape(-1,1)).detach().numpy()
        else:
            new_fake = g(batch[:,0].reshape(-1,1)).detach().numpy()
            fakes = np.concatenate((fakes,new_fake))
    return fakes    
    


#--------------
# Train Process
#--------------
def Train(dataloader,
          noiseloader,
          g,
          d,
          loss_F,
          epochs = 200,
          momentum = [0.3, 0.7],
          echo=True):
    
    """Loss_F should be function that can be called"""
    
    mom1 = momentum[0]
    mom2 = momentum[1]
    
    flag = 1
    best_flag = 1
    
    d_loss = []
    g_loss = []
    
    optimizer_G, optimizer_D = Optimizer(g, d) 
    
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            
            # Adversarial ground truths
            valid = Variable(torch.Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)    
            
            # ----------------------------------------------------------------
            # Train Generator
            optimizer_G.zero_grad()
        
            # Sample noise as generator input
            z = Variable(torch.Tensor([np.random.normal(size=(1)) for _ in range(batch.size(0))]))
            gen_output = g(z)
        
            # measure loss for generator
            gen_loss = loss_F(d(gen_output), valid)
            
            if gen_loss<=2:
                if best_flag==1:
                    best_fakes = GenerateFake(noiseloader, g)
                    best_flag += 1
                else:
                    best_fakes = mom1 * best_fakes + mom2 * GenerateFake(noiseloader, g)
            
            # backpropagete gen_loss and optimize
            gen_loss.backward()
            optimizer_G.step()
            # ----------------------------------------------------------------
            
            
            # ----------------------------------------------------------------
            # Train Discriminator
            optimizer_D.zero_grad()
    
            
            # Measure Discriminator Loss
            d_real_loss = loss_F(d(batch),valid)
            d_fake_loss = loss_F(d(gen_output.detach()),fake)
            dis_loss = (d_real_loss+d_fake_loss)/2
            
            dis_loss.backward()
            optimizer_D.step()
            
            g_loss.append(gen_loss)
            d_loss.append(dis_loss)
            
            if ((epoch*16)+(i+1))%100==0:
                
                if echo==True:
                    print('Epoch: ', epoch+1, ' Batch: ',i+1)
                    print("[D loss: {}] [G loss: {}]".format(dis_loss, gen_loss))
                
                if epoch>=20:
                    if flag==1:
                        fakes = GenerateFake(noiseloader, g)
                        flag += 1
                    else:
                        fakes = mom1 * fakes + mom2 * GenerateFake(noiseloader, g)          
            
            # ----------------------------------------------------------------
            
    return fakes, best_fakes, g.eval(), d.eval(), g_loss, d_loss