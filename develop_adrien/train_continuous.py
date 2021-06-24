#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:35:11 2021

@author: adrienbitton
"""




import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import soundfile as sf
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from newLSTMpreprocess import pctof,mtof

from ops_continuous import load_and_pp_data,sample_minibatch,export_minibatch,plot_losses,gradient_check,ModelContinuousPitch_FandA



def print_time(s_duration):
    m,s = divmod(s_duration,60)
    h, m = divmod(m, 60)
    return h, m, s



###############################################################################
## settings

# default DDSP setting is frame_rate=100 and sample_rate=16000

# CUDA_VISIBLE_DEVICES=0 python train_continuous.py --mname test00 --batch_size 8 --train_len 1000 --N_iter 50000 --hidden_size 512
# 1000 iterationns ~ 3"30

# TODO: check warning asking to apply flatten_parameters() on RNN (which ? ddsp's ?)

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("RUNNING DEVICE IS ",device)

parser = argparse.ArgumentParser()
# global setting
parser.add_argument('--mname',    type=str,   default="test")
parser.add_argument('--data_dir',    type=str,   default='/fast-2/adrien/intership-expressive-DDSP/dataset/dataset.pickle')
parser.add_argument('--ddsp_path',    type=str,   default='/fast-2/adrien/intership-expressive-DDSP/results/ddsp_debug_pretrained.ts')
# training setting
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--train_len',    type=int,   default=1000)
parser.add_argument('--N_iter',    type=int,   default=50000)
parser.add_argument('--lr',    type=float,   default=0.0001)
parser.add_argument('--f0_weight',    type=float,   default=10.)
parser.add_argument('--export_step',    type=int,   default=1000)
parser.add_argument('--test_percent',    type=float,   default=0.2)
# model setting
parser.add_argument('--articulation_percent',    type=float,   default=0.1)
parser.add_argument('--hidden_size',    type=int,   default=256)
parser.add_argument('--out_n_bins',    type=int,   default=64)
args = parser.parse_args()
print(args)



###############################################################################
## initialisation

train_data,test_data,f0_range,loudness_range = load_and_pp_data(args.data_dir,args.train_len,articulation_percent=args.articulation_percent,test_percent=args.test_percent)

# hard-coded
input_size = 4
out_size = 2
model = ModelContinuousPitch_FandA(input_size, args.hidden_size, f0_range, loudness_range, n_out=out_size, n_bins=args.out_n_bins, f0_weight=args.f0_weight, ddsp_path=args.ddsp_path)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)


mpath = "./"+args.mname+"/"
export_dir = mpath+'export/'
os.makedirs(mpath)
os.makedirs(export_dir)

dict_args = vars(args)
dict_args["f0_range"] = f0_range
dict_args["loudness_range"] = loudness_range
np.save(os.path.join(mpath,args.mname+'_args.npy'),dict_args)


mb_input,mb_target = sample_minibatch(train_data,args.batch_size,args.train_len,device)
gradient_check(model,optimizer,mb_input,mb_target)



###############################################################################
## training

start_time = timeit.default_timer()

train_losses = {"loss_tot":[],"loss_f0_frame":[],"loss_f0_articulation":[],"loss_loudness_frame":[],"loss_loudness_articulation":[]}
test_losses = {"loss_tot":[],"loss_f0_frame":[],"loss_f0_articulation":[],"loss_loudness_frame":[],"loss_loudness_articulation":[]}
step_losses = []

train_losslog = {"loss_tot":0,"loss_f0_frame":0,"loss_f0_articulation":0,"loss_loudness_frame":0,"loss_loudness_articulation":0}
test_losslog = {"loss_tot":0,"loss_f0_frame":0,"loss_f0_articulation":0,"loss_loudness_frame":0,"loss_loudness_articulation":0}
tr_count = 0


for __i in range(1,args.N_iter+1):
    
    # training step
    model.train()
    mb_input,mb_target = sample_minibatch(train_data,args.batch_size,args.train_len,device)
    losses = model.forward_loss(mb_input,mb_target)
    loss_tot = sum(losses)
    model.zero_grad()
    optimizer.zero_grad()
    loss_tot.backward()
    optimizer.step()
    train_losslog["loss_tot"] += loss_tot.item()
    train_losslog["loss_f0_frame"] += losses[0].item()
    train_losslog["loss_f0_articulation"] += losses[1].item()
    train_losslog["loss_loudness_frame"] += losses[2].item()
    train_losslog["loss_loudness_articulation"] += losses[3].item()
    tr_count += 1
    
    # test step
    model.eval()
    with torch.no_grad():
        mb_input,mb_target = sample_minibatch(test_data,args.batch_size,args.train_len,device)
        losses = model.forward_loss(mb_input,mb_target)
        loss_tot = sum(losses)
    test_losslog["loss_tot"] += loss_tot.item()
    test_losslog["loss_f0_frame"] += losses[0].item()
    test_losslog["loss_f0_articulation"] += losses[1].item()
    test_losslog["loss_loudness_frame"] += losses[2].item()
    test_losslog["loss_loudness_articulation"] += losses[3].item()
    
    if __i%100==0:
        step_losses.append(__i)
        h, m, s = print_time(timeit.default_timer()-start_time)
        for k in train_losslog:
            train_losslog[k] /= tr_count
            test_losslog[k] /= tr_count
        print("\n"+args.mname+'  elapsed time = '+"%d:%02d:%02d" % (h, m, s)+'  iter '+str(__i)+'   out of #iters '+str(args.N_iter))
        print('training losses',train_losslog)
        print('test losses',test_losslog)
        for k in train_losslog:
            train_losses[k].append(train_losslog[k])
            test_losses[k].append(test_losslog[k])
            train_losslog[k] = 0
            test_losslog[k] = 0
        tr_count = 0
    
    if __i%args.export_step==0:
        print("intermediate plots and exports")
        mb_input,mb_target = sample_minibatch(train_data,args.batch_size,args.train_len,device)
        export_minibatch(model,device,mb_input,mb_target,export_dir+"train_",sample_rate=16000)
        mb_input,mb_target = sample_minibatch(test_data,args.batch_size,args.train_len,device)
        export_minibatch(model,device,mb_input,mb_target,export_dir+"test_",sample_rate=16000)
        plot_losses(train_losses,step_losses,export_dir+"train_")
        plot_losses(test_losses,step_losses,export_dir+"test_")



###############################################################################
## happy ending

torch.save(model.state_dict(), mpath+args.mname+".pt")
mb_input,mb_target = sample_minibatch(train_data,args.batch_size,args.train_len,device)
export_minibatch(model,device,mb_input,mb_target,export_dir+"train_",sample_rate=16000)
mb_input,mb_target = sample_minibatch(test_data,args.batch_size,args.train_len,device)
export_minibatch(model,device,mb_input,mb_target,export_dir+"test_",sample_rate=16000)
plot_losses(train_losses,step_losses,export_dir+"train_")
plot_losses(test_losses,step_losses,export_dir+"test_")


mb_input,mb_target = sample_minibatch(train_data,args.batch_size,args.train_len,device)
gradient_check(model,optimizer,mb_input,mb_target)




