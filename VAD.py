#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:38:51 2023
@author: sariyanide
"""
import cv2
import os
import torch
import scipy.signal
torch.manual_seed(1957)
from time import time


import argparse
import numpy as np
from sys import exit
from utils import compute_features_from_lmks, CNN_VAD

parser = argparse.ArgumentParser()
parser.add_argument('--file_lmks', type=str, required=True)
parser.add_argument('--file_txt_out', type=str, required=True)
parser.add_argument('--file_video_in', type=str, required=False, default=None)
parser.add_argument('--file_video_out', type=str, required=False, default=None)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--mode', type=str, default='convolve')
parser.add_argument('--model_file', type=str, default='./models/cnn1d-w30.pth')
parser.add_argument('--convolve_rate', type=float, default=0.4)
parser.add_argument('--xlabel', type=int, default=-1)
parser.add_argument('--ylabel', type=int, default=-1)

args = parser.parse_args()

if os.path.exists(args.file_txt_out):
    print(f'Skipping because the output file already exisits: {args.file_txt_out}')
    exit(0)

if not os.path.exists(os.path.dirname(args.file_txt_out)):
    print(f'Skipping because output directory does not exist at {os.path.dirname(args.file_txt_out)}')
    exit(1)

args.use_std = 0
model_type = os.path.basename(args.model_file).split('-')[0]
checkpoint = torch.load(f'{args.model_file}', map_location=torch.device('cpu'))
mp = checkpoint['model_params']

args.use_all_lmks = mp['use_all_lmks']
args.use_std = mp['use_std']
args.nparts = mp['nparts']
args.w = mp['w']

if 'zm_feats' not in mp:
    mp['zm_feats'] = 0

K0 = mp['base_filters']
stride = mp['stride']
model = CNN_VAD(mp['in_features'], mp['kernel_size'], mp['kernel_size2'], mp['base_filters'], mp['dropout_rate'], mp['stride'], mp['nparts'])

model = model.to(args.device)
model.network.load_state_dict(checkpoint['model_state'])


def get_processed_data_test(lmks_file):
    lmks = np.loadtxt(lmks_file)
    xraw = []
    for i in range(lmks.shape[0]-args.w):
        xraw.append(lmks[i:(i+args.w),:])
    
    xraw = np.array(xraw)
    
    X = []
    for i in range(xraw.shape[0]):
        xi = xraw[i,:,:]
        X.append(compute_features_from_lmks(xi, mp['w'], mp['zm_feats'], mp['nparts'], mp['use_all_lmks'], mp['use_std']))
    
    X = np.array(X)
    
    return torch.from_numpy(X).float()

x = get_processed_data_test(args.file_lmks).to(args.device)

model.eval()
y = model(x).argmax(axis=1).cpu().numpy()
if args.mode == 'direct':
    y = np.pad(y, (int(args.w/2), int(args.w/2)), mode='constant')
    s = scipy.signal.medfilt(y, 5)
    
elif args.mode == 'convolve':
    f = np.ones((args.w,))
    s = np.convolve(y,f)

Nframes = s.shape[0]


def add_label(img, text,
          pos=(100, 920),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=4,
          text_color=(0, 0, 255),
          text_color_bg=(255, 255, 255)
          ):

    
    if label.lower() == 'speaking':
        text_color = (0, 255, 0)
    
    pad = 10
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    #x -= text_w//2
    y += int(text_h*1.5)
    cv2.rectangle(img, (x-pad, y-pad), (x + text_w+pad, y + text_h+pad), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size



write_output_video = (args.file_video_in is not None) and (args.file_video_out is not None)

if write_output_video:
    # Read the video file
    video_capture = cv2.VideoCapture(args.file_video_in)

    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    # Get the video's frame width, height, and frames per second
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = int(video_capture.get(5))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(args.file_video_out+'.avi', fourcc, fps, (frame_width, frame_height))


#%%


ix = 0
output = []
for frame_id in range(Nframes):
    # while True:
    if write_output_video:
        ret, frame = video_capture.read()

        if not ret:
            break
    
    label = ''
    is_speaking = (args.mode == 'direct' and s[frame_id] == 1) or (args.mode == 'convolve' and s[frame_id] > args.convolve_rate*args.w)
    
    output.append(int(is_speaking))
    
    if not write_output_video:
        continue
    
    if ix < len(s):
        if is_speaking:
            label = 'SPEAKING'
        else:
            label = 'SILENT'
            
    if args.xlabel == -1 or args.ylabel == -1:
        pos = (10, frame_height//2)
    else:
        pos = (args.xlabel, args.ylabel)
    
    # Add the label to the current frame
    add_label(frame, label, pos)
    output_video.write(frame)

    """
    # Display the frame with the label
    cv2.imshow('Video with Label', frame)

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """



np.savetxt(args.file_txt_out, output, fmt='%d')

if write_output_video:
    # Release the video capture object and close the display window
    video_capture.release()
    cv2.destroyAllWindows()
    output_video.release()
    
    os.system('ffmpeg -i %s.avi %s 2> /dev/null' % (args.file_video_out, args.file_video_out))
    os.system('rm %s.avi' % args.file_video_out)
    





