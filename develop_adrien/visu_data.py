#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:53:38 2021

@author: adrienbitton
"""


import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pickle

from newLSTMpreprocess import pctof


# TODO: check how to segment frame and articulation when midi pitches are contiguous
# TODO: check why max(e_loudness) < max(u_loudness) ??

# onsets are events==+1 and offsets are events==-1

# if offset, u_f0 drops to 0 --> articulation segment
# if onset without offset, take a region around and make u_f0 = 0 to create articulation segment


pp_data = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/dataset/dataset.pickle'

with (open(pp_data, "rb")) as openfile:
    data = pickle.load(openfile)

N_features = len(data.keys())
fnames = list(data.keys())
x_start = 0
# x_start = 16000
# x_start = 20000
x_range = 500


# plot all features

plt.figure()
for i in range(N_features):
    plt.subplot(N_features+1,1,i+1)
    # plt.title(fnames[i],loc="right")
    plt.title(fnames[i],x=1.07,y=0.3)
    if fnames[i]=="e_cents":
        plt.plot(data[fnames[i]][x_start:x_start+x_range]-0.5)
    else:
        plt.plot(data[fnames[i]][x_start:x_start+x_range])
    plt.xticks([])
    plt.xlim((0, x_range)) 
plt.subplot(N_features+1,1,i+2)
# plt.title("e_f0+e_cents",loc="right")
plt.title("pctof(e_f0,e_cents-0.5)",x=1.07,y=0.3) # cf newLSTMpreprocess.py --> cents are shifted to [0,1] before saving the pickle dataset !!!
plt.plot(pctof(data["e_f0"][x_start:x_start+x_range],data["e_cents"][x_start:x_start+x_range]-0.5))
plt.xlim((0, x_range)) 
# plt.tight_layout()
plt.show()



# plot expressive f0 and put color based on frame or articulation

articulation_percent = 0.1

def segment_midi_input(data,articulation_percent = 0.1):
    event_pos = np.where(data["events"]!=0)[0]
    u_f0_segment = data["u_f0"].copy()
    
    # zeroing all events
    for crt_pos in event_pos:
        u_f0_segment[crt_pos] = 0
    
    # first onset
    u_f0_segment[event_pos[0]:event_pos[0]+round((event_pos[1]-event_pos[0])*articulation_percent)] = 0
    
    # all intermediate events
    for i in range(1,len(event_pos)-1):
        prv_pos = event_pos[i-1]
        crt_pos = event_pos[i]
        nxt_pos = event_pos[i+1]
        u_f0_segment[crt_pos-round((crt_pos-prv_pos)*articulation_percent):crt_pos] = 0
        u_f0_segment[crt_pos:crt_pos+round((nxt_pos-crt_pos)*articulation_percent)] = 0
    
    # last offset
    u_f0_segment[event_pos[-1]-round((event_pos[-1]-event_pos[-2])*articulation_percent):event_pos[-1]] = 0
    return u_f0_segment

u_f0_segment = segment_midi_input(data,articulation_percent = articulation_percent)

articulation_pos = np.where(u_f0_segment==0)[0]
frame_pos = np.where(u_f0_segment!=0)[0]

min_u_loud = np.min(data["u_loudness"])
cut_u_loud = int(min_u_loud-3)
u_loudness_segment = data["u_loudness"].copy()
u_loudness_segment[articulation_pos] = cut_u_loud

f0_articulation = pctof(data["e_f0"],data["e_cents"]-0.5).copy()
f0_frame = pctof(data["e_f0"],data["e_cents"]-0.5).copy()
f0_articulation[frame_pos] = 0
f0_frame[articulation_pos] = 0

cents_articulation = data["e_cents"].copy()-0.5
cents_frame = data["e_cents"].copy()-0.5
cents_articulation[frame_pos] = -10
cents_frame[articulation_pos] = -10

min_loud = np.min(data["e_loudness"])
cut_loud = int(min_loud-3)
loudness_articulation = data["e_loudness"].copy()
loudness_frame = data["e_loudness"].copy()
loudness_articulation[frame_pos] = cut_loud
loudness_frame[articulation_pos] = cut_loud


plt.figure()
plt.subplot(6,1,1)
plt.title("u_f0",x=1.07,y=0.3)
plt.plot(data["u_f0"][x_start:x_start+x_range],c="r")
plt.plot(u_f0_segment[x_start:x_start+x_range],c="b")
plt.xlim((0, x_range))
plt.ylim(np.min(data["u_f0"][x_start:x_start+x_range])-3,np.max(data["u_f0"][x_start:x_start+x_range])+3)
plt.xticks([])

plt.subplot(6,1,2)
plt.title("u_loudness",x=1.07,y=0.3)
plt.plot(data["u_loudness"][x_start:x_start+x_range],c="r")
plt.plot(u_loudness_segment[x_start:x_start+x_range],c="b")
plt.xlim((0, x_range))
plt.ylim(min_u_loud-0.5,0.5)
plt.xticks([])

plt.subplot(6,1,3)
plt.title("events",x=1.07,y=0.3)
plt.plot(data["events"][x_start:x_start+x_range])
plt.xlim((0, x_range))
plt.xticks([])

plt.subplot(6,1,4)
plt.title("f0",x=1.07,y=0.3)
plt.plot(f0_frame,c="b")
plt.plot(f0_articulation,c="r")
plt.xlim((x_start, x_start+x_range))
plt.ylim(np.min(pctof(data["e_f0"],data["e_cents"]-0.5)[x_start:x_start+x_range])-100,np.max(pctof(data["e_f0"],data["e_cents"]-0.5)[x_start:x_start+x_range])+100)
plt.xticks([])

plt.subplot(6,1,5)
plt.title("cents",x=1.07,y=0.3)
plt.plot(cents_frame,c="b")
plt.plot(cents_articulation,c="r")
plt.xlim((x_start, x_start+x_range))
plt.ylim(-0.6,0.6)
plt.xticks([])

plt.subplot(6,1,6)
plt.title("loudness",x=1.07,y=0.3)
plt.plot(loudness_frame,c="b")
plt.plot(loudness_articulation,c="r")
plt.xlim((x_start, x_start+x_range))
plt.ylim(np.min(data["e_loudness"][x_start:x_start+x_range])-0.5,np.max(data["e_loudness"][x_start:x_start+x_range])+0.5)
plt.xticks([])
plt.show()





