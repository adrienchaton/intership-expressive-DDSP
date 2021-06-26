#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:00:20 2021

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

from ops_continuous import load_and_pp_data,sample_minibatch,generate_minibatch,reconstruct_minibatch,plot_losses,gradient_check,ModelContinuousPitch_FandA



device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrain_dir = "./outputs/"
data_dir = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/dataset/dataset.pickle'
ddsp_path = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/results/ddsp_debug_pretrained.ts'

mnames = ["test00","test01_1bin","test02_1bin_minmax","test03_1bin_quantile"][:]

for mname in mnames:
    mpath = pretrain_dir+mname+"/"
    export_dir = mpath+"play0/"
    try:
        os.makedirs(export_dir)
    except:
        pass
    
    args = mpath+mname+"_args.npy"
    args = np.load(args,allow_pickle=True).item()
    
    args["batch_size"] = 2
    
    train_data,test_data,f0_range,loudness_range,scalers = load_and_pp_data(data_dir,args["train_len"],articulation_percent=args["articulation_percent"],test_percent=args["test_percent"],scaler=args["scaler"])  
    
    # hard-coded
    input_size = 4
    out_size = 2
    try:
        model = ModelContinuousPitch_FandA(input_size, args["hidden_size"], f0_range, loudness_range, n_out=out_size, n_bins=args["out_n_bins"], f0_weight=args["f0_weight"], ddsp_path=ddsp_path, n_RNN=args["n_RNN"], dp=args["dp"])         
    except:
        model = ModelContinuousPitch_FandA(input_size, args["hidden_size"], f0_range, loudness_range, n_out=out_size, n_bins=args["out_n_bins"], f0_weight=args["f0_weight"], ddsp_path=ddsp_path)  
    model.load_state_dict(torch.load(os.path.join(mpath,mname+'.pt'), map_location='cpu'))
    model.eval()
    model.to(device)
    
    # if scalers is not None:
        # TODO: save scalers
    model.scalers = scalers
    
    # plot end loss
    
    train_losses = np.load(mpath+"export/train_losses.npy",allow_pickle=True).item()
    test_losses = np.load(mpath+"export/test_losses.npy",allow_pickle=True).item()
    
    n_steps = len(train_losses["loss_tot"])
    step_losses = list(np.arange(n_steps))[n_steps//2:]
    
    for k in train_losses.keys():
        train_losses[k] = train_losses[k][n_steps//2:]
        test_losses[k] = test_losses[k][n_steps//2:]
    
    plot_losses(train_losses,step_losses,export_dir+"train_end_")
    plot_losses(test_losses,step_losses,export_dir+"test_end_")
    
    # export
    
    mb_input,mb_target = sample_minibatch(train_data,args["batch_size"],args["train_len"],device)
    generate_minibatch(model,device,mb_input,mb_target,export_dir+"train_",sample_rate=16000)
    reconstruct_minibatch(model,device,mb_input,mb_target,export_dir+"train_",sample_rate=16000)
    
    mb_input,mb_target = sample_minibatch(test_data,args["batch_size"],args["train_len"],device)
    generate_minibatch(model,device,mb_input,mb_target,export_dir+"test_",sample_rate=16000)
    reconstruct_minibatch(model,device,mb_input,mb_target,export_dir+"test_",sample_rate=16000)




