# -*- coding: utf-8 -*-
"""

@author: wnchang
"""

# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset



class FLUX(nn.Module):
    def __init__(self, matrix, n_modules, f_in = 50, f_out = 1):
        super(FLUX, self).__init__()
        # gene to flux
        self.inSize = f_in
        
        
        self.m_encoder = nn.ModuleList([
                                        nn.Sequential(nn.Linear(self.inSize,8, bias = False),
                                                      nn.Tanhshrink(),
                                                      nn.Linear(8, f_out),
                                                      nn.Tanhshrink()
                                                      )
                                        for i in range(n_modules)])
        
                        
                                        
    

    
    def updateC(self, m, n_comps, cmMat): # stochastic matrix
        #cmMat = torch.empty(12, 14).random_(2)       
        c = torch.zeros((m.shape[0], n_comps))
        for i in range(c.shape[1]):
            tmp = m * cmMat[i,:]
            c[:,i] = torch.sum(tmp, dim=1)
        
        return c
       
    
    def scale_m(self, out_m_batch, scale_vector, cell_id_batch):
        
        out_m_batch_new = out_m_batch
        batchSize = out_m_batch.shape[0]
        if batchSize == 1:
            out_m_batch_new = out_m_batch_new * scale_vector[cell_id_batch]
        else:
            for i in range(batchSize):
                out_m_batch_new[i,:] = out_m_batch_new[i,:] * scale_vector[cell_id_batch[i]]
        
        return out_m_batch_new
    

    def forward(self, x, n_modules, n_genes, n_comps, cmMat, scale, cell_id):
#        x = torch.transpose(x, 0, 1)
        
        for i in range(n_modules):
            x_block = x[:,i*n_genes: (i+1)*n_genes,]
            subnet = self.m_encoder[i]
            if i == 0:
                m = subnet(x_block) 
            else:
                m = torch.cat((m, subnet(x_block)),1)

        

        m = self.scale_m(m, scale, cell_id)

        c = self.updateC(m, n_comps, cmMat)

        
        return m, c
    