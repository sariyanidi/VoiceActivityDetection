#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:25:31 2023

@author: sariyanide
"""
import os
import random
import numpy as np
from scipy.stats import skew, kurtosis
from glob import glob

random.seed(1907)

# we assume that the raw data is in the folder below. 
# Download from the following link: blob:https://sariyanidi.com/5bf61a4a-b8c0-4406-80a4-985a5fdfbae1 
srcdir = './data/VAD_train_data/raw'

dst_dir = './data/VAD_train_data'
os.makedirs(dst_dir, exist_ok=True)

pfiles = glob(f'{srcdir}/pos_class/*canonical_lmks') 
nfiles = glob(f'{srcdir}/neg_class/*canonical_lmks') 

Nptra = int(0.75*len(pfiles))
Nntra = int(0.75*len(nfiles))

w = 30
all_files_tra = pfiles[0:Nptra]+nfiles[0:Nntra]
random.shuffle(all_files_tra)

all_files_tes = pfiles[Nptra:]+nfiles[Nntra:]
random.shuffle(all_files_tes)

def create_data_matrices(all_files, extend_short=True):
    X = []
    labels = []
    for f in all_files:
        x = np.loadtxt(f)
        
        if x.shape[0] < int(w/2.)+1:
            continue
        
        if x.shape[0] < w:
            if not extend_short:
                continue
            x = np.concatenate((x, np.flip(x, axis=0)))
            
        xs = np.split(x, np.arange(w,x.shape[0],w), axis=0)[:-1]
        xs.append(x[-w:,:])
        if f.find('pos_class') > 0:
            label = 1
        else:
            label = 0
            
        for x in xs:
            if np.sum(np.isnan(np.mean(x))) > 0 or np.sum(np.isnan(np.std(x))) > 0 or np.sum(np.isnan(skew(x))) > 0 or np.sum(np.isnan(kurtosis(x))) > 0:
                continue
            
            X.append(x)
            labels.append(label)
            
    Y = np.array(labels).reshape(-1,1)
    X = np.array(X)
    
    return X, Y
    
expand = 1
    
(Xtra, Ytra) = create_data_matrices(all_files_tra, bool(expand))
(Xtes, Ytes) = create_data_matrices(all_files_tes, False)

N = len(pfiles)+len(nfiles)

np.save(f'./{dst_dir}/Xtra-%d-%d.npy' % (expand, w), Xtra)
np.save(f'./{dst_dir}/Ytra-%d-%d.npy' % (expand, w), Ytra)
np.save(f'./{dst_dir}/Xtes-%d-%d.npy' %  (expand, w), Xtes)
np.save(f'./{dst_dir}/Ytes-%d-%d.npy' %  (expand, w), Ytes)

expand = 0
    
(Xtra, Ytra) = create_data_matrices(all_files_tra, bool(expand))
(Xtes, Ytes) = create_data_matrices(all_files_tes, False)

np.save(f'./{dst_dir}/Xtra-%d-%d.npy' % (expand, w), Xtra)
np.save(f'./{dst_dir}/Ytra-%d-%d.npy' % (expand, w), Ytra)
np.save(f'./{dst_dir}/Xtes-%d-%d.npy' %  (expand, w), Xtes)
np.save(f'./{dst_dir}/Ytes-%d-%d.npy' %  (expand, w), Ytes)


print(f'N={N}')
