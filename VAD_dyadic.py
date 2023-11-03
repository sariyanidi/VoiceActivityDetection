#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:09:53 2023

@author: sariyanide
"""

import os
import numpy as np
import torch
from glob import glob
import argparse

def get_text_pos(rects_file):
    rects = np.loadtxt(rects_file)
    x0s = rects[0,:]
    y0s = rects[1,:]
    xfs = x0s+rects[2,:]
    yfs = y0s+rects[3,:]
    
    xc = (np.median(x0s)+np.median(xfs))/2.0
    yc = np.median(yfs)
    
    pos_str = f'--xlabel={int(xc)} --ylabel={int(yc)}'
    
    return pos_str

parser = argparse.ArgumentParser()
parser.add_argument('lmks1_path', type=str)
parser.add_argument('lmks2_path', type=str)
parser.add_argument('vid1_path', type=str)
parser.add_argument('vid2_path', type=str)
parser.add_argument('video_out_path', type=str)

parser.add_argument('--rects1_path', type=str, default=None)
parser.add_argument('--rects2_path', type=str, default=None)
parser.add_argument('--mode', type=str, default='convolve')
parser.add_argument('--convolve_rate', type=float, default=0.4)
parser.add_argument('--model_fpath', type=str, default='./models/cnn1d-w30.pth')

args = parser.parse_args()

l1 = args.lmks1_path
l2 = args.lmks2_path
vi1 = args.vid1_path
vi2 = args.vid2_path

rects1_path = args.rects1_path
rects2_path = args.rects2_path

w = torch.load(args.model_fpath)['model_params']['w']
bn_model = '.'.join(os.path.basename(args.model_fpath).split('.')[:-1])

pos_str1 = ''
pos_str2 = ''

if rects1_path is not None and rects2_path is not None \
    and os.path.exists(rects1_path) and os.path.exists(rects2_path):
    pos_str1 = get_text_pos(rects1_path)
    pos_str2 = get_text_pos(rects2_path)    

tail = f'-{args.mode}-{bn_model}'
if args.mode == 'convolve':
    tail += f'-{args.convolve_rate:.2}'
    
vo1 = f'./tmp1{os.path.basename(vi1)}.mp4'
vo2 = f'./tmp2{os.path.basename(vi2)}.mp4'

cmd1 = f'python VAD.py --file_lmks={l1} --file_video_in={vi1} --file_video_out={vo1} --model_file={args.model_fpath}'\
    f' --convolve_rate={args.convolve_rate} --mode={args.mode} {pos_str1}'
cmd2 = f'python VAD.py --file_lmks={l2} --file_video_in={vi2} --file_video_out={vo2} --model_file={args.model_fpath}'\
    f' --convolve_rate={args.convolve_rate} --mode={args.mode} {pos_str2}'
cmdc = f'ffmpeg -i {vo1} -i {vo2} -filter_complex hstack=inputs=2 {args.video_out_path}'

if not os.path.exists(args.video_out_path):
    print(cmd1)
    os.system(cmd1)
    
    print(cmd2)
    os.system(cmd2)
    
    print(cmdc)
    os.system(cmdc)
    
    print(f'\tOutput written to video: {args.video_out_path}')
    
os.system(f'rm {vo1} {vo2}')

