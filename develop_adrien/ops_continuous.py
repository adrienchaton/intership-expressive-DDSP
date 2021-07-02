#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:01:11 2021

@author: adrienbitton
"""



import sys
import os
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pickle
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from newLSTMpreprocess import pctof,mtof


##############################################################################
## utils

def segment_midi_input(data, articulation_percent = 0.1):
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


def make_raw_f0(data):
    raw_f0 = pctof(data["e_f0"],data["e_cents"])
    return raw_f0


def split_data(data,train_len,test_percent = 0.2):
    dataset_len = len(data["u_f0"])
    assert int(dataset_len*test_percent)>train_len, "test set length is too short"
    train_data = dict()
    train_data["u_f0"] = data["u_f0"][:int(dataset_len*(1-test_percent))]
    train_data["u_loudness"] = data["u_loudness"][:int(dataset_len*(1-test_percent))]
    train_data["raw_f0"] = data["raw_f0"][:int(dataset_len*(1-test_percent))]
    train_data["e_loudness"] = data["e_loudness"][:int(dataset_len*(1-test_percent))]
    train_data["u_f0_segment"] = data["u_f0_segment"][:int(dataset_len*(1-test_percent))]
    test_data = dict()
    test_data["u_f0"] = data["u_f0"][int(dataset_len*(1-test_percent)):]
    test_data["u_loudness"] = data["u_loudness"][int(dataset_len*(1-test_percent)):]
    test_data["raw_f0"] = data["raw_f0"][int(dataset_len*(1-test_percent)):]
    test_data["e_loudness"] = data["e_loudness"][int(dataset_len*(1-test_percent)):]
    test_data["u_f0_segment"] = data["u_f0_segment"][int(dataset_len*(1-test_percent)):]
    return train_data,test_data


def load_and_pp_data(file, train_len, articulation_percent = 0.1, test_percent = 0.2, scaler = "none"):
    with (open(file, "rb")) as openfile:
        data = pickle.load(openfile)
    u_f0_segment = segment_midi_input(data, articulation_percent = articulation_percent)
    raw_f0 = make_raw_f0(data)
    data["u_f0_segment"] = u_f0_segment
    data["raw_f0"] = raw_f0
    f0_range = [np.min(raw_f0),np.max(raw_f0)]
    loudness_range = [np.min(data["e_loudness"]),np.max(data["e_loudness"])]
    if scaler=="none":
        scalers = None
    # compute scalers for u_f0,u_loudness,raw_f0,e_loudness
    # preprocess all dataset and apply inverse tranform when exporting --> losses are computed in the scaled range
    elif scaler=="quantile":
        scaler_u_f0 = QuantileTransformer(n_quantiles=64,output_distribution='uniform')
        data["u_f0"] = scaler_u_f0.fit_transform(data["u_f0"].reshape(-1, 1)).reshape(-1)
        scaler_u_loudness = QuantileTransformer(n_quantiles=64,output_distribution='uniform')
        data["u_loudness"] = scaler_u_loudness.fit_transform(data["u_loudness"].reshape(-1, 1)).reshape(-1)
        scaler_raw_f0 = QuantileTransformer(n_quantiles=64,output_distribution='uniform')
        data["raw_f0"] = scaler_raw_f0.fit_transform(data["raw_f0"].reshape(-1, 1)).reshape(-1)
        scaler_e_loudness = QuantileTransformer(n_quantiles=64,output_distribution='uniform')
        data["e_loudness"] = scaler_e_loudness.fit_transform(data["e_loudness"].reshape(-1, 1)).reshape(-1)
        scalers = {"scaler_u_f0":scaler_u_f0,"scaler_u_loudness":scaler_u_loudness,
                   "scaler_raw_f0":scaler_raw_f0,"scaler_e_loudness":scaler_e_loudness}
    elif scaler=="minmax":
        scaler_u_f0 = MinMaxScaler()
        data["u_f0"] = scaler_u_f0.fit_transform(data["u_f0"].reshape(-1, 1)).reshape(-1)
        scaler_u_loudness = MinMaxScaler()
        data["u_loudness"] = scaler_u_loudness.fit_transform(data["u_loudness"].reshape(-1, 1)).reshape(-1)
        scaler_raw_f0 = MinMaxScaler()
        data["raw_f0"] = scaler_raw_f0.fit_transform(data["raw_f0"].reshape(-1, 1)).reshape(-1)
        scaler_e_loudness = MinMaxScaler()
        data["e_loudness"] = scaler_e_loudness.fit_transform(data["e_loudness"].reshape(-1, 1)).reshape(-1)
        scalers = {"scaler_u_f0":scaler_u_f0,"scaler_u_loudness":scaler_u_loudness,
                   "scaler_raw_f0":scaler_raw_f0,"scaler_e_loudness":scaler_e_loudness}
    else:
        print("invalid scaler argument")
    
    train_data,test_data = split_data(data,train_len,test_percent=test_percent)
    return train_data,test_data,f0_range,loudness_range,scalers


def sample_minibatch(data,batch_size,train_len,device,model_prediction="AR"):
    # for AR model (auto-regressive)
    # input = u_f0,u_loudness + raw_f0[-1],e_loudness[-1] + u_f0_segment[-1]
    # target = raw_f0,e_loudness + u_f0_segment
    
    # for FF model (feed-forward)
    # input = u_f0,u_loudness + u_f0_segment
    # target = raw_f0,e_loudness
    
    dataset_len = len(data["u_f0"])
    mb_input = []
    mb_target = []
    for i in range(batch_size):
        rand_start = np.random.choice(np.arange(1,dataset_len-train_len))
        
        if model_prediction=="AR":
            mb_input.append(np.stack([data["u_f0"][rand_start:rand_start+train_len],
                                      data["u_loudness"][rand_start:rand_start+train_len],
                                      data["raw_f0"][rand_start-1:rand_start+train_len-1],
                                      data["e_loudness"][rand_start-1:rand_start+train_len-1],
                                      data["u_f0_segment"][rand_start-1:rand_start+train_len-1]],-1))
            mb_target.append(np.stack([data["raw_f0"][rand_start:rand_start+train_len],
                                       data["e_loudness"][rand_start:rand_start+train_len],
                                       data["u_f0_segment"][rand_start:rand_start+train_len]],-1))
        
        if model_prediction=="FF":
            mb_input.append(np.stack([data["u_f0"][rand_start:rand_start+train_len],
                                      data["u_loudness"][rand_start:rand_start+train_len],
                                      data["u_f0_segment"][rand_start:rand_start+train_len]],-1))
            mb_target.append(np.stack([data["raw_f0"][rand_start:rand_start+train_len],
                                       data["e_loudness"][rand_start:rand_start+train_len],
                                       data["u_f0_segment"][rand_start:rand_start+train_len]],-1))
    
    mb_input = torch.from_numpy(np.stack(mb_input,0)).float().to(device)
    mb_target = torch.from_numpy(np.stack(mb_target,0)).float().to(device)
    return mb_input,mb_target


def generate_minibatch(model,device,mb_input,mb_target,path,sample_rate=16000,model_prediction="AR"):
    batch_size = mb_input.shape[0]
    for i in range(batch_size):
        with torch.no_grad():
            u_f0 = mb_input[i,:,:1]
            u_loudness = mb_input[i,:,1:2]
            u_f0_segment = mb_target[i,:,2:]
            if model_prediction=="AR":
                init_f0 = mb_input[i,0,2]
                init_loudness = mb_input[i,0,3]
                output_f0,output_loudness = model.generation_loop(u_f0,u_loudness,u_f0_segment,device,init_f0=init_f0,init_loudness=init_loudness)
            if model_prediction=="FF":
                output_f0,output_loudness = model.generation_loop(u_f0,u_loudness,u_f0_segment,device)
            
            raw_f0 = mb_target[i,:,:1]
            e_loudness = mb_target[i,:,1:2]
            
            if model.scalers is not None:
                u_f0 = torch.from_numpy(model.scalers["scaler_u_f0"].inverse_transform(u_f0.cpu().numpy())).float().to(device)
                u_loudness = torch.from_numpy(model.scalers["scaler_u_loudness"].inverse_transform(u_loudness.cpu().numpy())).float().to(device)
                output_f0 = torch.from_numpy(model.scalers["scaler_raw_f0"].inverse_transform(output_f0.unsqueeze(-1).cpu().numpy())).squeeze(-1).float().to(device)
                output_loudness = torch.from_numpy(model.scalers["scaler_e_loudness"].inverse_transform(output_loudness.unsqueeze(-1).cpu().numpy())).squeeze(-1).float().to(device)
                
                raw_f0 = torch.from_numpy(model.scalers["scaler_raw_f0"].inverse_transform(raw_f0.cpu().numpy())).float().to(device)
                e_loudness = torch.from_numpy(model.scalers["scaler_e_loudness"].inverse_transform(e_loudness.cpu().numpy())).float().to(device)
            
            output_audio = model.get_audio(output_f0, output_loudness)
            target_audio = model.get_audio(raw_f0.squeeze(-1), e_loudness.squeeze(-1))
        
        sf.write(path+str(i)+"_gen_output_audio.wav",output_audio.squeeze().cpu().numpy(),sample_rate)
        sf.write(path+str(i)+"_gen_target_audio.wav",target_audio.squeeze().cpu().numpy(),sample_rate)
        
        plt.figure(figsize=(12,8))
        plt.suptitle("u_f0 / u_loudness / target f0 / output f0 / target loudness / output loudness")
        plt.subplot(6,1,1)
        plt.plot(u_f0.squeeze().cpu().numpy(),c="r")
        plt.plot(u_f0_segment.squeeze().cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,2)
        plt.plot(u_loudness.squeeze().cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,3)
        plt.plot(raw_f0.squeeze(-1).cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,4)
        plt.plot(output_f0.squeeze().cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,5)
        plt.plot(e_loudness.squeeze(-1).cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,6)
        plt.plot(output_loudness.squeeze().cpu().numpy(),c="b")
        # plt.xticks([])
        plt.savefig(path+str(i)+"_gen_controls.pdf")
        plt.close("all")


def reconstruct_minibatch(model,device,mb_input,mb_target,path,sample_rate=16000):
    with torch.no_grad():
        prediction = model.forward(mb_input)
        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = model.split_predictions(prediction)
        u_f0_segment = mb_target[:,:,2]
        output_f0 = torch.where(u_f0_segment==0,pred_f0_articulation,pred_f0_frame)
        output_loudness = torch.where(u_f0_segment==0,pred_loudness_articulation,pred_loudness_frame)
    
    batch_size = mb_input.shape[0]
    for i in range(batch_size):
        u_f0 = mb_input[i,:,:1]
        u_loudness = mb_input[i,:,1:2]
        u_f0_segment = mb_target[i,:,2:]
        
        raw_f0 = mb_target[i,:,:1]
        e_loudness = mb_target[i,:,1:2]
        
        if model.scalers is not None:
            u_f0 = torch.from_numpy(model.scalers["scaler_u_f0"].inverse_transform(u_f0.cpu().numpy())).float().to(device)
            u_loudness = torch.from_numpy(model.scalers["scaler_u_loudness"].inverse_transform(u_loudness.cpu().numpy())).float().to(device)
            output_f0[i,:] = torch.from_numpy(model.scalers["scaler_raw_f0"].inverse_transform(output_f0[i,:].unsqueeze(-1).cpu().numpy())).squeeze(-1).float().to(device)
            output_loudness[i,:] = torch.from_numpy(model.scalers["scaler_e_loudness"].inverse_transform(output_loudness[i,:].unsqueeze(-1).cpu().numpy())).squeeze(-1).float().to(device)
            
            raw_f0 = torch.from_numpy(model.scalers["scaler_raw_f0"].inverse_transform(raw_f0.cpu().numpy())).float().to(device)
            e_loudness = torch.from_numpy(model.scalers["scaler_e_loudness"].inverse_transform(e_loudness.cpu().numpy())).float().to(device)
        
        with torch.no_grad():
            output_audio = model.get_audio(output_f0[i,:], output_loudness[i,:])
            target_audio = model.get_audio(raw_f0.squeeze(-1), e_loudness.squeeze(-1))
        
        sf.write(path+str(i)+"_rec_output_audio.wav",output_audio.squeeze().cpu().numpy(),sample_rate)
        sf.write(path+str(i)+"_rec_target_audio.wav",target_audio.squeeze().cpu().numpy(),sample_rate)
        
        plt.figure(figsize=(12,8))
        plt.suptitle("u_f0 / u_loudness / target f0 / output f0 / target loudness / output loudness")
        plt.subplot(6,1,1)
        plt.plot(u_f0.squeeze().cpu().numpy(),c="r")
        plt.plot(u_f0_segment.squeeze().cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,2)
        plt.plot(u_loudness.squeeze().cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,3)
        plt.plot(raw_f0.squeeze(-1).cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,4)
        plt.plot(output_f0[i,:].cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,5)
        plt.plot(e_loudness.squeeze(-1).cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,6)
        plt.plot(output_loudness[i,:].cpu().numpy(),c="b")
        # plt.xticks([])
        plt.savefig(path+str(i)+"_rec_controls.pdf")
        plt.close("all")


def plot_losses(losses,step_losses,path):
    losses_names = list(losses.keys())
    plt.figure(figsize=(12,8))
    plt.suptitle(str(losses_names))
    for i,lname in enumerate(losses_names):
        plt.subplot(len(losses_names),1,i+1)
        plt.plot(step_losses,losses[lname])
        if i<len(losses_names)-1:
            plt.xticks([])
    plt.savefig(path+"losses.pdf")
    plt.close("all")


def gradient_check(model,optimizer,mb_input,mb_target):
    print("\n\nforwarding tot_loss")
    losses = model.forward_loss(mb_input,mb_target)
    model.zero_grad()
    optimizer.zero_grad()
    tot_loss = sum(losses)
    tot_loss.backward()
    tot_grad = 0
    named_p = model.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    print("\n\nforwarding loss_f0_frame")
    loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = model.forward_loss(mb_input,mb_target)
    model.zero_grad()
    optimizer.zero_grad()
    loss_f0_frame.backward()
    tot_grad = 0
    named_p = model.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    print("\n\nforwarding loss_f0_articulation")
    loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = model.forward_loss(mb_input,mb_target)
    model.zero_grad()
    optimizer.zero_grad()
    loss_f0_articulation.backward()
    tot_grad = 0
    named_p = model.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    print("\n\nforwarding loss_loudness_frame")
    loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = model.forward_loss(mb_input,mb_target)
    model.zero_grad()
    optimizer.zero_grad()
    loss_loudness_frame.backward()
    tot_grad = 0
    named_p = model.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)
    
    print("\n\nforwarding loss_loudness_articulation")
    loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = model.forward_loss(mb_input,mb_target)
    model.zero_grad()
    optimizer.zero_grad()
    loss_loudness_articulation.backward()
    tot_grad = 0
    named_p = model.named_parameters()
    for name, param in named_p:
        if param.grad is not None:
            sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
            if sum_abs_paramgrad==0:
                print(name,'sum_abs_paramgrad==0')
            else:
                tot_grad += sum_abs_paramgrad
        else:
            print(name,'param.grad is None')
    print('tot_grad = ',tot_grad)


##############################################################################
## auto-regressive model class

## input of both Frame and Articulation networks is u_f0,u_loudness,raw_f0[-1],e_loudness[-1]
## output of both Frame and Articulation networks is raw_f0,e_loudness

# def exponential_sigmoid(x):
#     return 2*F.sigmoid(x)**np.log(10)+1e-7

class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=True, act=True):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size) if norm else None
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act:
            x = nn.functional.leaky_relu(x)
        return x

class ModelContinuousPitch_FandA_AR(nn.Module):
    def __init__(self, in_size, hidden_size, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path="results/ddsp_debug_pretrained.ts",
                 n_RNN=1, dp=0.):#, scalers):
        super().__init__()
        
        # for each network (Frame or Articulation)
        # in_size = 4 for u_f0,u_loudness,raw_f0,e_loudness
        # out_size = n_out*n_bins -> n_out=2 after weighted sum -> raw_f0,e_loudness
        
        self.n_bins = n_bins
        if self.n_bins>1:
            # f0 is tiled in log and loudness is tiled in linear
            f0_bins = np.logspace(np.log10(f0_range[0]),np.log10(f0_range[1]),num=n_bins,endpoint=True,base=10)
            self.f0_bins = torch.nn.parameter.Parameter(torch.from_numpy(f0_bins).float(), requires_grad=False)
            loudness_bins = np.linspace(loudness_range[0],loudness_range[1],num=n_bins,endpoint=True)
            self.loudness_bins = torch.nn.parameter.Parameter(torch.from_numpy(loudness_bins).float(), requires_grad=False)
        
        self.f0_weight = f0_weight
        
        # self.save_hyperparameters()
        # self.scalers = scalers
        # self.loudness_nbins = 30
        self.ddsp = torch.jit.load(ddsp_path).eval()
        
        
        ## frame modules
        F_pre_lstm = [LinearBlock(in_size, hidden_size)]
        if dp>0:
            F_pre_lstm += [nn.Dropout(p=dp)]
        F_pre_lstm += [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            F_pre_lstm += [nn.Dropout(p=dp)]
        self.F_pre_lstm = nn.Sequential(*F_pre_lstm)

        self.F_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_RNN,
            dropout=dp,
            batch_first=True,
        )
        
        F_post_lstm = [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            F_post_lstm += [nn.Dropout(p=dp)]
        F_post_lstm += [LinearBlock(hidden_size,n_out*n_bins,norm=False,act=False)]
        self.F_post_lstm = nn.Sequential(*F_post_lstm)
        
        
        ## articulation modules
        A_pre_lstm = [LinearBlock(in_size, hidden_size)]
        if dp>0:
            A_pre_lstm += [nn.Dropout(p=dp)]
        A_pre_lstm += [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            A_pre_lstm += [nn.Dropout(p=dp)]
        self.A_pre_lstm = nn.Sequential(*A_pre_lstm)

        self.A_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_RNN,
            dropout=dp,
            batch_first=True,
        )
        
        A_post_lstm = [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            A_post_lstm += [nn.Dropout(p=dp)]
        A_post_lstm += [LinearBlock(hidden_size,n_out*n_bins,norm=False,act=False)]
        self.A_post_lstm = nn.Sequential(*A_post_lstm)

    # def configure_optimizers(self):
    #     return torch.optim.Adam(
    #         self.parameters(),
    #         lr=1e-4,
    #         weight_decay=.01,
    #     )

    def forward(self, x):
        batch_size = x.shape[0]
        frame_size = x.shape[1]
        # x should be [batch,frames,5] == u_f0,u_loudness,raw_f0,e_loudness + u_f0_segment
        x_in = x[:,:,:-1]
        u_f0_segment = x[:,:,-1]
        
        # forward frame network
        F_out = self.F_pre_lstm(x_in)
        F_out = self.F_lstm(F_out)[0]
        F_out = self.F_post_lstm(F_out)
        if self.n_bins>1:
            # weighted sum of bins
            pred_f0_frame = torch.sum(F.softmax(F_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
            pred_loudness_frame = torch.sum(F.softmax(F_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        else:
            pred_f0_frame = F_out[:,:,0]
            pred_loudness_frame = F_out[:,:,1]
            if self.scalers is not None:
                pred_f0_frame = F.sigmoid(pred_f0_frame)
                pred_loudness_frame = F.sigmoid(pred_loudness_frame)
        
        # replace pred_f0_frame into ground-truth f0 and pred_loudness_frame into ground-truth loudness
        # ground-truth is one step behind predictions -> ground-truth[1:] and predictions[:-1]
        x_in[:,1:,2] = torch.where(u_f0_segment[:,1:]==0,x_in[:,1:,2],pred_f0_frame[:,:-1].detach()) # where is ==0 is ground-truth articulation otherwise is predicted frame (and detached)
        x_in[:,1:,3] = torch.where(u_f0_segment[:,1:]==0,x_in[:,1:,3],pred_loudness_frame[:,:-1].detach())
        
        # forward articulation network conditioned on frame prediction
        A_out = self.A_pre_lstm(x_in)
        A_out = self.A_lstm(A_out)[0]
        A_out = self.A_post_lstm(A_out)
        if self.n_bins>1:
            # weighted sum of bins
            pred_f0_articulation = torch.sum(F.softmax(A_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
            pred_loudness_articulation = torch.sum(F.softmax(A_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        else:
            pred_f0_articulation = A_out[:,:,0]
            pred_loudness_articulation = A_out[:,:,1]
            if self.scalers is not None:
                pred_f0_articulation = F.sigmoid(pred_f0_articulation)
                pred_loudness_articulation = F.sigmoid(pred_loudness_articulation)
        
        prediction = torch.stack([pred_f0_frame,pred_loudness_frame,pred_f0_articulation,pred_loudness_articulation],-1)
        return prediction

    def split_predictions(self, prediction):
        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = torch.split(prediction, 1, -1)
        return pred_f0_frame.squeeze(-1), pred_loudness_frame.squeeze(-1), pred_f0_articulation.squeeze(-1), pred_loudness_articulation.squeeze(-1)

    def weighted_mse_loss(self, pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation,
                   mb_target):
        target_f0, target_loudness, u_f0_segment = torch.split(mb_target, 1, -1)
        target_f0, target_loudness, u_f0_segment = target_f0.squeeze(-1), target_loudness.squeeze(-1), u_f0_segment.squeeze(-1)

        loss_f0_frame = nn.functional.mse_loss(torch.where(u_f0_segment!=0,pred_f0_frame,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment!=0,target_f0,torch.zeros_like(pred_f0_frame)))*self.f0_weight
        loss_f0_articulation = nn.functional.mse_loss(torch.where(u_f0_segment==0,pred_f0_articulation,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment==0,target_f0,torch.zeros_like(pred_f0_frame)))*self.f0_weight
        
        loss_loudness_frame = nn.functional.mse_loss(torch.where(u_f0_segment!=0,pred_loudness_frame,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment!=0,target_loudness,torch.zeros_like(pred_f0_frame)))
        loss_loudness_articulation = nn.functional.mse_loss(torch.where(u_f0_segment==0,pred_loudness_articulation,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment==0,target_loudness,torch.zeros_like(pred_f0_frame)))
        
        return loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation

    def forward_loss(self, mb_input,mb_target):
        prediction = self.forward(mb_input)

        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = self.split_predictions(prediction)
        
        loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = self.weighted_mse_loss(pred_f0_frame,
                                                    pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation,mb_target)

        return loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation

    # def sample_one_hot(self, x):
    #     n_bin = x.shape[-1]
    #     sample = torch.distributions.Categorical(logits=x).sample()
    #     sample = nn.functional.one_hot(sample, n_bin)
    #     return sample

    @torch.no_grad()
    def generation_loop(self, u_f0,u_loudness,u_f0_segment,device,init_f0=None,init_loudness=None):
        if len(u_f0.shape)==2:
            u_f0 = u_f0.unsqueeze(0)
            u_loudness = u_loudness.unsqueeze(0)
            u_f0_segment = u_f0_segment.unsqueeze(0)
        batch_size = u_f0.shape[0]
        assert batch_size==1, "generation loop should be run on an individual track"
        frame_size = u_f0.shape[1]
        
        # u_f0,u_loudness,u_f0_segment should be aligned
        # init_f0,init_loudness should be the previous step or it is set from u_f0,u_loudness
        x = torch.cat([u_f0,u_loudness,torch.zeros_like(u_f0),torch.zeros_like(u_f0)],-1)
        # x should be [batch,frames,4] == u_f0,u_loudness,raw_f0,e_loudness
        x = torch.cat([x,torch.zeros(batch_size,1,4).to(device)],1) # additional col for the last output step
        
        if init_f0 is not None:
            x[0,0,2] = init_f0
        else:
            x[0,0,2] = mtof(x[0,0,0])
        
        if init_loudness is not None:
            x[0,0,3] = init_loudness
        else:
            x[0,0,3] = x[0,0,1]
        
        F_context = None
        A_context = None

        for i in range(frame_size):
            x_in = x[:, i:i + 1, :]
            
            # forward frame network
            F_out = self.F_pre_lstm(x_in)
            F_out,F_context = self.F_lstm(F_out,F_context)
            F_out = self.F_post_lstm(F_out)
            
            # forward articulation network
            A_out = self.A_pre_lstm(x_in)
            A_out,A_context = self.A_lstm(A_out,A_context)
            A_out = self.A_post_lstm(A_out)
            
            # select either frame or articulation prediction
            if u_f0_segment[0,i]!=0:
                if self.n_bins>1:
                    # weighted sum of bins
                    pred_f0 = torch.sum(F.softmax(F_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0),2)
                    pred_loudness = torch.sum(F.softmax(F_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0),2)
                else:
                    pred_f0 = F_out[:,:,0]
                    pred_loudness = F_out[:,:,1]
                    if self.scalers is not None:
                        pred_f0 = F.sigmoid(pred_f0)
                        pred_loudness = F.sigmoid(pred_loudness)
            else:
                if self.n_bins>1:
                    # weighted sum of bins
                    pred_f0 = torch.sum(F.softmax(A_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0),2)
                    pred_loudness = torch.sum(F.softmax(A_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0),2)
                else:
                    pred_f0 = A_out[:,:,0]
                    pred_loudness = A_out[:,:,1]
                    if self.scalers is not None:
                        pred_f0 = F.sigmoid(pred_f0)
                        pred_loudness = F.sigmoid(pred_loudness)
            
            x[:, i + 1:i + 2, 2] = pred_f0
            x[:, i + 1:i + 2, 3] = pred_loudness

        output_f0 = x[0,1:,2]
        output_loudness = x[0,1:,3]

        return output_f0,output_loudness

    # def apply_inverse_transform(self, x, idx):
    #     scaler = self.scalers[idx]
    #     x = x.cpu()
    #     out = scaler.inverse_transform(x.reshape(-1, 1))
    #     out = torch.from_numpy(out).to("cuda")
    #     out = out.unsqueeze(0)
    #     return out.float()

    def get_audio(self, f0, loudness):
        if len(f0.shape)==1:
            f0 = f0.unsqueeze(0).unsqueeze(-1)
            loudness = loudness.unsqueeze(0).unsqueeze(-1)
        y = self.ddsp(f0, loudness)
        return y

    # def validation_step(self, batch, batch_idx):
    #     model_input, target = batch
    #     prediction = self.forward(model_input.float())

    #     pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
    #     target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

    #     loss_f0, loss_cents, loss_loudness = self.mse_and_ce(
    #         pred_f0,
    #         pred_cents,
    #         pred_loudness,
    #         target_f0,
    #         target_cents,
    #         target_loudness,
    #     )

    #     self.log("val_loss_f0", loss_f0)
    #     self.log("val_loss_cents", loss_cents)
    #     self.log("val_loss_loudness", loss_loudness)
    #     self.log("val_total", loss_f0 + loss_cents + loss_loudness)

    #     ## Every 100 epochs : produce audio

    #     if self.current_epoch % 20 == 0:

    #         audio = self.get_audio(model_input[0], target[0])
    #         # output audio in Tensorboard
    #         tb = self.logger.experiment
    #         n = "Epoch={}".format(self.current_epoch)
    #         tb.add_audio(tag=n, snd_tensor=audio, sample_rate=16000)

    #         # TODO : CHANGE THIS VISUALISATION BUG
    #         for i in range(model_input.shape[1]):
    #             tb.add_scalars("f0", {
    #                 "model": model_input[0, i, 0],
    #                 "target": target[0, i, 0],
    #             }, i)
    #             tb.add_scalars("cents", {
    #                 "model": model_input[0, i, 1],
    #                 "target": target[0, i, 1]
    #             }, i)



##############################################################################
## feed-forward model class

## input of Frame network is u_f0,u_loudness
## input of Articulation network is u_f0,u_loudness + raw_f0,e_loudness (frame prediction)

## output of both Frame and Articulation networks is raw_f0,e_loudness

class ModelContinuousPitch_FandA_FF(nn.Module):
    def __init__(self, in_size, hidden_size, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path="results/ddsp_debug_pretrained.ts",
                 n_dir=1, n_RNN=1, dp=0.):#, scalers):
        super().__init__()
        
        # for Frame network
        # in_size//2 = 2 for u_f0,u_loudness
        
        # for Articulation network
        # in_size = 4 for u_f0,u_loudness,raw_f0,e_loudness
        
        # out_size = n_out*n_bins -> n_out=2 after weighted sum -> raw_f0,e_loudness
        
        self.n_dir = n_dir
        if n_dir==2:
            bidirectional = True
        else:
            bidirectional = False
        
        self.f0_range = f0_range
        self.loudness_range = loudness_range
        
        self.n_bins = n_bins
        if self.n_bins>1:
            # f0 is tiled in log and loudness is tiled in linear
            f0_bins = np.logspace(np.log10(f0_range[0]),np.log10(f0_range[1]),num=n_bins,endpoint=True,base=10)
            self.f0_bins = torch.nn.parameter.Parameter(torch.from_numpy(f0_bins).float(), requires_grad=False)
            loudness_bins = np.linspace(loudness_range[0],loudness_range[1],num=n_bins,endpoint=True)
            self.loudness_bins = torch.nn.parameter.Parameter(torch.from_numpy(loudness_bins).float(), requires_grad=False)
        
        self.f0_weight = f0_weight
        
        self.ddsp = torch.jit.load(ddsp_path).eval()
        
        
        ## frame modules
        F_pre_lstm = [LinearBlock(in_size//2, hidden_size)]
        if dp>0:
            F_pre_lstm += [nn.Dropout(p=dp)]
        F_pre_lstm += [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            F_pre_lstm += [nn.Dropout(p=dp)]
        self.F_pre_lstm = nn.Sequential(*F_pre_lstm)

        self.F_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_RNN,
            dropout=dp,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        if n_dir==2:
            F_post_lstm = [LinearBlock(2*hidden_size, hidden_size)]
        else:
            F_post_lstm = [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            F_post_lstm += [nn.Dropout(p=dp)]
        F_post_lstm += [LinearBlock(hidden_size,n_out*n_bins,norm=False,act=False)]
        self.F_post_lstm = nn.Sequential(*F_post_lstm)
        
        
        ## articulation modules
        A_pre_lstm = [LinearBlock(in_size, hidden_size)]
        if dp>0:
            A_pre_lstm += [nn.Dropout(p=dp)]
        A_pre_lstm += [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            A_pre_lstm += [nn.Dropout(p=dp)]
        self.A_pre_lstm = nn.Sequential(*A_pre_lstm)

        self.A_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=n_RNN,
            dropout=dp,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        if n_dir==2:
            A_post_lstm = [LinearBlock(2*hidden_size, hidden_size)]
        else:
            A_post_lstm = [LinearBlock(hidden_size, hidden_size)]
        if dp>0:
            A_post_lstm += [nn.Dropout(p=dp)]
        A_post_lstm += [LinearBlock(hidden_size,n_out*n_bins,norm=False,act=False)]
        self.A_post_lstm = nn.Sequential(*A_post_lstm)

    def forward(self, x):
        batch_size = x.shape[0]
        frame_size = x.shape[1]
        # x should be [batch,frames,3] == u_f0,u_loudness + u_f0_segment (aligned)
        x_in = x[:,:,:-1]
        u_f0_segment = x[:,:,-1]
        
        # forward frame network
        F_out = self.F_pre_lstm(x_in)
        F_out = self.F_lstm(F_out)[0]
        F_out = self.F_post_lstm(F_out)
        if self.n_bins>1:
            # weighted sum of bins
            pred_f0_frame = torch.sum(F.softmax(F_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
            pred_loudness_frame = torch.sum(F.softmax(F_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        else:
            pred_f0_frame = F_out[:,:,0]
            pred_loudness_frame = F_out[:,:,1]
            if self.scalers is not None:
                pred_f0_frame = F.sigmoid(pred_f0_frame)
                pred_loudness_frame = F.sigmoid(pred_loudness_frame)
        
        # stack pred_f0_frame and pred_loudness_frame
        if self.scalers is None:
            x_frame = torch.stack([torch.where(u_f0_segment!=0,pred_f0_frame.detach(),torch.zeros_like(pred_f0_frame)+self.f0_range[0]),
                                   torch.where(u_f0_segment!=0,pred_loudness_frame.detach(),torch.zeros_like(pred_loudness_frame)+self.loudness_range[0])],-1)
        else:
            # assume scaling in [0,1]
            x_frame = torch.stack([torch.where(u_f0_segment!=0,pred_f0_frame.detach(),torch.zeros_like(pred_f0_frame)),
                                   torch.where(u_f0_segment!=0,pred_loudness_frame.detach(),torch.zeros_like(pred_loudness_frame))],-1)
        x_in = torch.cat([x_in,x_frame],-1)
        
        # forward articulation network conditioned on frame prediction
        A_out = self.A_pre_lstm(x_in)
        A_out = self.A_lstm(A_out)[0]
        A_out = self.A_post_lstm(A_out)
        if self.n_bins>1:
            # weighted sum of bins
            pred_f0_articulation = torch.sum(F.softmax(A_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
            pred_loudness_articulation = torch.sum(F.softmax(A_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        else:
            pred_f0_articulation = A_out[:,:,0]
            pred_loudness_articulation = A_out[:,:,1]
            if self.scalers is not None:
                pred_f0_articulation = F.sigmoid(pred_f0_articulation)
                pred_loudness_articulation = F.sigmoid(pred_loudness_articulation)
        
        prediction = torch.stack([pred_f0_frame,pred_loudness_frame,pred_f0_articulation,pred_loudness_articulation],-1)
        return prediction

    def split_predictions(self, prediction):
        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = torch.split(prediction, 1, -1)
        return pred_f0_frame.squeeze(-1), pred_loudness_frame.squeeze(-1), pred_f0_articulation.squeeze(-1), pred_loudness_articulation.squeeze(-1)

    def weighted_mse_loss(self, pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation,
                   mb_target):
        target_f0, target_loudness, u_f0_segment = torch.split(mb_target, 1, -1)
        target_f0, target_loudness, u_f0_segment = target_f0.squeeze(-1), target_loudness.squeeze(-1), u_f0_segment.squeeze(-1)

        loss_f0_frame = nn.functional.mse_loss(torch.where(u_f0_segment!=0,pred_f0_frame,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment!=0,target_f0,torch.zeros_like(pred_f0_frame)))*self.f0_weight
        loss_f0_articulation = nn.functional.mse_loss(torch.where(u_f0_segment==0,pred_f0_articulation,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment==0,target_f0,torch.zeros_like(pred_f0_frame)))*self.f0_weight
        
        loss_loudness_frame = nn.functional.mse_loss(torch.where(u_f0_segment!=0,pred_loudness_frame,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment!=0,target_loudness,torch.zeros_like(pred_f0_frame)))
        loss_loudness_articulation = nn.functional.mse_loss(torch.where(u_f0_segment==0,pred_loudness_articulation,torch.zeros_like(pred_f0_frame)),
                                               torch.where(u_f0_segment==0,target_loudness,torch.zeros_like(pred_f0_frame)))
        
        return loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation

    def forward_loss(self, mb_input,mb_target):
        prediction = self.forward(mb_input)

        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = self.split_predictions(prediction)
        
        loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation = self.weighted_mse_loss(pred_f0_frame,
                                                    pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation,mb_target)

        return loss_f0_frame,loss_f0_articulation,loss_loudness_frame,loss_loudness_articulation

    @torch.no_grad()
    def generation_loop(self, u_f0,u_loudness,u_f0_segment,device):
        if len(u_f0.shape)==2:
            u_f0 = u_f0.unsqueeze(0)
            u_loudness = u_loudness.unsqueeze(0)
            u_f0_segment = u_f0_segment.unsqueeze(0)
        batch_size = u_f0.shape[0]
        assert batch_size==1, "generation loop should be run on an individual track"
        
        # u_f0,u_loudness,u_f0_segment should be aligned
        x = torch.cat([u_f0,u_loudness,u_f0_segment],-1)
        
        prediction = self.forward(x)
        pred_f0_frame, pred_loudness_frame, pred_f0_articulation, pred_loudness_articulation = self.split_predictions(prediction)
        
        output_f0 = torch.where(u_f0_segment[:,:,0]==0,pred_f0_articulation,pred_f0_frame)
        output_loudness = torch.where(u_f0_segment[:,:,0]==0,pred_loudness_articulation,pred_loudness_frame)

        return output_f0.squeeze(),output_loudness.squeeze()

    def get_audio(self, f0, loudness):
        if len(f0.shape)==1:
            f0 = f0.unsqueeze(0).unsqueeze(-1)
            loudness = loudness.unsqueeze(0).unsqueeze(-1)
        y = self.ddsp(f0, loudness)
        return y


"""
##############################################################################
## test

# torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/dataset/dataset.pickle'
ddsp_path = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/results/ddsp_debug_pretrained.ts'

batch_size = 2
train_len = 1000

model_prediction = "FF"

train_data,test_data,f0_range,loudness_range,scalers = load_and_pp_data(file,train_len, articulation_percent = 0.1, test_percent = 0.2)

N_features = len(train_data.keys())
fnames = list(train_data.keys())

if model_prediction=="AR":
    model = ModelContinuousPitch_FandA_AR(4, 128, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path=ddsp_path, n_RNN=2, dp=0.)
if model_prediction=="FF":
    model = ModelContinuousPitch_FandA_FF(4, 128, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path=ddsp_path, n_dir=2, n_RNN=2, dp=0.)

model.train()
model.to(device)

model.scalers = scalers

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

mb_input,mb_target = sample_minibatch(train_data,batch_size,train_len,device,model_prediction=model_prediction)

## checking gradient

gradient_check(model,optimizer,mb_input,mb_target)

## checking generation loop and audio synthesis

generate_minibatch(model,device,mb_input,mb_target,"./train_",sample_rate=16000,model_prediction=model_prediction)

reconstruct_minibatch(model,device,mb_input,mb_target,"./train_",sample_rate=16000)
"""


