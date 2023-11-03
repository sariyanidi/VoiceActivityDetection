#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:38:51 2023
@author: sariyanide
"""

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, fbeta_score
from torch.nn import functional as F
import os
import torch
torch.manual_seed(1957)

from utils import compute_features_from_lmks, CNN_VAD

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_size2', type=int, default=1) # 1
parser.add_argument('--base_filters', type=int, default=96)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--scheduler_gamma', type=float, default=0.95)
parser.add_argument('--scheduler_step_size', type=int, default=600)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--nparts', type=int, default=10)
parser.add_argument('--zm_feats', type=int, default=1)
parser.add_argument('--w', type=int, default=30)
parser.add_argument('--use_all_lmks', type=int, default=0)
parser.add_argument('--n_tot_epochs', type=int, default=800)
parser.add_argument('--plot_every', type=int, default=50)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--use_std', type=int, default=1)


args = parser.parse_args()
key = f'CNN-z{args.zm_feats}-{args.kernel_size}-{args.kernel_size2}-{args.base_filters}-{args.learning_rate}-{args.stride}' \
        f'-d{args.dropout_rate}-ua{args.use_all_lmks}{args.scheduler_gamma}-{args.scheduler_step_size}-{args.w}'
            
if args.nparts != args.w:
    key += f'-{args.nparts}'

if args.device.find('cuda') == -1:
    key += 'cpu'

if args.use_std:
    key += 'STD'

print(args)
device = args.device

def get_processed_data(expand, w, is_tra=True):
    if is_tra:
        Xraw = np.load('./data/VAD_train_data/Xtra-%d-%d.npy' % (int(expand), w))
        Y = np.load('./data/VAD_train_data/Ytra-%d-%d.npy' % (int(expand), w))
    else:
        Xraw = np.load('./data/VAD_train_data/Xtes-%d-%d.npy' % (int(expand), w))
        Y = np.load('./data/VAD_train_data/Ytes-%d-%d.npy' % (int(expand), w))

    w = Xraw.shape[1]
    
    X = []
    for i in range(Xraw.shape[0]):
        x = Xraw[i,:,:]
        X.append(compute_features_from_lmks(x, args.w, args.zm_feats, args.nparts, args.use_all_lmks, args.use_std))
    
    X = np.array(X)
    
    return torch.from_numpy(X).float(), torch.from_numpy(Y).long()

Xtra, Ytra = get_processed_data(True, args.w, True)
Xtes, Ytes = get_processed_data(True, args.w, False)

Xtes = Xtes.to(device)
Ytes = Ytes.to(device)
K = Xtra.shape[-1]

Ntra = Ytra.shape[0]
Ntes = Ytes.shape[0]
Ntot = Ntra + Ntes

batch_size = 128

Xbatches = Xtra.split(batch_size)
Ybatches = Ytra.split(batch_size)
hist_tra = []
hist_tes = []
f1s = []
balanced_accs = []
accs = []

nb_epochs_finished = 0
model_params = {'in_features': Xtra.shape[-2],
                'kernel_size': args.kernel_size,
                'kernel_size2': args.kernel_size2,
                'use_all_lmks': args.use_all_lmks,
                'dropout_rate': args.dropout_rate,
                'base_filters': args.base_filters,
                'use_std': args.use_std,
                'nparts': args.nparts,
                'zm_feats': args.zm_feats,
                'w': args.w,
                'stride': args.stride}

K0 = args.base_filters 
model = CNN_VAD(Xtra.shape[-2], args.kernel_size, args.kernel_size2, args.base_filters, args.dropout_rate, args.stride, Xtra.shape[-1])

checkpoint_dir = './models/checkpoints' 
figures_dir = './models/figures'

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

checkpoint_file = f'{checkpoint_dir}/cnn1d-{key}.pth'

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    if args.device.find('cuda') > -1:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    else:
        torch.set_rng_state(checkpoint["rng_state"])
    
    model.network.load_state_dict(checkpoint["model_state"])
    hist_tra = checkpoint['hist_tra']
    hist_tes = checkpoint['hist_tes']
    balanced_accs = checkpoint['balanced_accs']
    accs = checkpoint['accs']
    nb_epochs_finished = checkpoint['nb_epochs_finished']
    print('Continuing from checkpoint -- done iterations: %d' % nb_epochs_finished)



model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.scheduler_step_size)

for n in range(nb_epochs_finished, args.n_tot_epochs):
    acc_train_loss = 0
    model.train()
    for batch_id in range(len(Xbatches)-1):
        inputs  = Xbatches[batch_id].to(device)
        targets = Ybatches[batch_id].to(device)
        
        output = model(inputs)
        loss = F.cross_entropy(output, targets.flatten())
        acc_train_loss += loss.item() #* inputs.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    hist_tra.append(acc_train_loss/len(Xbatches))
    
    model.eval()
    with torch.no_grad():
        output = model(Xtes)
        loss = F.cross_entropy(output, Ytes.flatten())
        hist_tes.append(loss.item())
        pred = output.argmax(dim=1).cpu().flatten().numpy()
        
        balanced_accs.append(balanced_accuracy_score(Ytes.cpu().numpy(), pred))
        f1s.append( fbeta_score(Ytes.cpu().numpy(), pred, beta=1.0) )
        accs.append(sum(pred  == Ytes.flatten().cpu().numpy())/len(pred))
        
    
    if n > 0 and (balanced_accs[-1] > max(balanced_accs[:-1])):
        checkpoint = {
            "model_state": model.network.state_dict(),
            "model_params": model_params,
            "balanced_accs": balanced_accs,
            "accs": accs,
            "f1s": f1s,
            "hist_tra": hist_tra,
            "hist_tes": hist_tes,
            "nb_epochs_finished": n,
        }
        
        if args.device.find('cuda') > -1:
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()
        else:
            checkpoint["rng_state"] = torch.get_rng_state()
            
        torch.save(checkpoint, checkpoint_file)

        
    if n % args.print_every == args.print_every-1:
        print('%4d, Balanced Accuracy: %.3f, Accuracy: %.3f' % (n+1, max(balanced_accs), max(accs)))
        
    if n % args.plot_every == args.plot_every-1:
        plt.clf()
        plt.subplot(211)
        plt.title(f'max_bal: {max(balanced_accs):.2f} -- max_F2: {max(f1s):.2f}')
        plt.semilogy(hist_tes)
        plt.subplot(212)
        plt.plot(accs)
        plt.plot(balanced_accs)
        plt.plot(f1s)
        plt.savefig(f'{figures_dir}/{key}.png')
        



