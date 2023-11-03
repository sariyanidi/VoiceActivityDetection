#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 09:04:04 2023

@author: sariyanide
"""
import numpy as np
import torch
import torch.nn as nn


def compute_features_from_lmks(x, w, zm_feats, nparts, use_all_lmks, use_std):
    wpart = int(w/nparts)
    
    if use_all_lmks == 0:
        x = x[:,93:]
        
    if zm_feats:
        x[:,::3] -= x[:,::3].mean()
        x[:,1::3] -= x[:,1::3].mean()
        x[:,2::3] -= x[:,2::3].mean()
    
    xsplit = np.array(np.split(x,range(wpart, w, wpart)))
    
    f = []
    f.append(np.mean(xsplit, axis=1))
    if use_std:
        f.append(np.std(xsplit, axis=1))
    
    f = np.array(f)
    f = np.transpose(f, (1,0,2))
    f = f.reshape(nparts, -1)
    f = np.transpose(f, (1,0))
    
    return f


class CNN_VAD(nn.Module):
    
    def __init__(self, in_features, kernel_size, kernel_size2, base_filters, dropout_rate, stride, seq_len):
        super().__init__(   )
        self.network = nn.Sequential(nn.BatchNorm1d(in_features),
                  nn.Conv1d(kernel_size=kernel_size, in_channels=in_features, out_channels=base_filters, stride=1),
                  nn.Dropout(dropout_rate),
                  nn.ReLU(),
                  nn.BatchNorm1d(base_filters), 
                  nn.Conv1d(kernel_size=kernel_size2, in_channels=base_filters, out_channels=base_filters, stride=stride),
                  nn.Dropout(dropout_rate),
                  nn.ReLU(),
                  nn.Flatten(),
                  nn.Linear((base_filters//stride)*(seq_len-(kernel_size-1)-(kernel_size2-1)), 2))
        
    def forward(self, x):
        return self.network(x)
        
        
