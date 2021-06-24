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


def load_and_pp_data(file, train_len, articulation_percent = 0.1, test_percent = 0.2):
    with (open(file, "rb")) as openfile:
        data = pickle.load(openfile)
    u_f0_segment = segment_midi_input(data, articulation_percent = articulation_percent)
    raw_f0 = make_raw_f0(data)
    data["u_f0_segment"] = u_f0_segment
    data["raw_f0"] = raw_f0
    f0_range = [np.min(raw_f0),np.max(raw_f0)]
    loudness_range = [np.min(data["e_loudness"]),np.max(data["e_loudness"])]
    train_data,test_data = split_data(data,train_len,test_percent=test_percent)
    return train_data,test_data,f0_range,loudness_range


def sample_minibatch(data,batch_size,train_len,device):
    # input = u_f0,u_loudness + raw_f0[-1],e_loudness[-1] + u_f0_segment[-1]
    # target = raw_f0,e_loudness + u_f0_segment
    dataset_len = len(data["u_f0"])
    mb_input = []
    mb_target = []
    for i in range(batch_size):
        rand_start = np.random.choice(np.arange(1,dataset_len-train_len))
        mb_input.append(np.stack([data["u_f0"][rand_start:rand_start+train_len],
                                  data["u_loudness"][rand_start:rand_start+train_len],
                                  data["raw_f0"][rand_start-1:rand_start+train_len-1],
                                  data["e_loudness"][rand_start-1:rand_start+train_len-1],
                                  data["u_f0_segment"][rand_start-1:rand_start+train_len-1]],-1))
        mb_target.append(np.stack([data["raw_f0"][rand_start:rand_start+train_len],
                                   data["e_loudness"][rand_start:rand_start+train_len],
                                   data["u_f0_segment"][rand_start:rand_start+train_len]],-1))
    mb_input = torch.from_numpy(np.stack(mb_input,0)).float().to(device)
    mb_target = torch.from_numpy(np.stack(mb_target,0)).float().to(device)
    return mb_input,mb_target


def export_minibatch(model,device,mb_input,mb_target,path,sample_rate=16000):
    batch_size = mb_input.shape[0]
    for i in range(batch_size):
        with torch.no_grad():
            u_f0 = mb_input[i,:,:1]
            u_loudness = mb_input[i,:,1:2]
            u_f0_segment = mb_target[i,:,2:]
            init_f0 = mb_input[i,0,2]
            init_loudness = mb_input[i,0,3]
            output_f0,output_loudness = model.generation_loop(u_f0,u_loudness,u_f0_segment,device,init_f0=init_f0,init_loudness=init_loudness)
            output_audio = model.get_audio(output_f0, output_loudness)
            target_audio = model.get_audio(mb_target[i,:,0], mb_target[i,:,1])
        
        sf.write(path+str(i)+"_output_audio.wav",output_audio.squeeze().cpu().numpy(),sample_rate)
        sf.write(path+str(i)+"_target_audio.wav",target_audio.squeeze().cpu().numpy(),sample_rate)
        
        plt.figure(figsize=(12,8))
        plt.subplot(6,1,1)
        plt.plot(u_f0.squeeze().cpu().numpy(),c="r")
        plt.plot(u_f0_segment.squeeze().cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,2)
        plt.plot(u_loudness.squeeze().cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,3)
        plt.plot(mb_target[i,:,0].cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,4)
        plt.plot(output_f0.squeeze().cpu().numpy(),c="b")
        plt.xticks([])
        plt.subplot(6,1,5)
        plt.plot(mb_target[i,:,1].cpu().numpy(),c="r")
        plt.xticks([])
        plt.subplot(6,1,6)
        plt.plot(output_loudness.squeeze().cpu().numpy(),c="b")
        # plt.xticks([])
        plt.savefig(path+str(i)+"_controls.pdf")
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
## nn

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

class ModelContinuousPitch_FandA(nn.Module):
    def __init__(self, in_size, hidden_size, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path="results/ddsp_debug_pretrained.ts"):#, scalers):
        super().__init__()
        
        # for each network (Frame or Articulation)
        # in_size = 4 for u_f0,u_loudness,raw_f0,e_loudness
        # out_size = n_out*n_bins -> n_out=2 after weighted sum -> raw_f0,e_loudness
        
        # f0 is tiled in log and loudness is tiled in linear
        self.n_bins = n_bins
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
        self.F_pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.F_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.F_post_lstm = nn.Sequential(
            LinearBlock(hidden_size, hidden_size),
            LinearBlock(
                hidden_size,
                n_out*n_bins,
                norm=False,
                act=False,
            ),
        )
        
        ## articulation modules
        self.A_pre_lstm = nn.Sequential(
            LinearBlock(in_size, hidden_size),
            LinearBlock(hidden_size, hidden_size),
        )

        self.A_lstm = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.A_post_lstm = nn.Sequential(
            LinearBlock(hidden_size, hidden_size),
            LinearBlock(
                hidden_size,
                n_out*n_bins,
                norm=False,
                act=False,
            ),
        )

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
        # weighted sum of bins
        pred_f0_frame = torch.sum(F.softmax(F_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        pred_loudness_frame = torch.sum(F.softmax(F_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        
        # replace pred_f0_frame into ground-truth f0 and pred_loudness_frame into ground-truth loudness
        # ground-truth is one step behind predictions -> ground-truth[1:] and predictions[:-1]
        x_in[:,1:,2] = torch.where(u_f0_segment[:,1:]==0,x_in[:,1:,2],pred_f0_frame[:,:-1].detach()) # where is ==0 is ground-truth articulation otherwise is predicted frame (and detached)
        x_in[:,1:,3] = torch.where(u_f0_segment[:,1:]==0,x_in[:,1:,3],pred_loudness_frame[:,:-1].detach())
        
        # forward articulation network conditioned on frame prediction
        A_out = self.A_pre_lstm(x_in)
        A_out = self.A_lstm(A_out)[0]
        A_out = self.A_post_lstm(A_out)
        # weighted sum of bins
        pred_f0_articulation = torch.sum(F.softmax(A_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        pred_loudness_articulation = torch.sum(F.softmax(A_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0).repeat(batch_size,frame_size,1),2)
        
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
            F_out,F_context = self.F_lstm(F_out)
            F_out = self.F_post_lstm(F_out)
            
            # forward articulation network conditioned on frame prediction
            A_out = self.A_pre_lstm(x_in)
            A_out,A_context = self.A_lstm(A_out)
            A_out = self.A_post_lstm(A_out)
            
            # weighted sum of bins
            if u_f0_segment[0,i]!=0:
                pred_f0 = torch.sum(F.softmax(F_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0),2)
                pred_loudness = torch.sum(F.softmax(F_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0),2)
            else:
                pred_f0 = torch.sum(F.softmax(A_out[:,:,:self.n_bins],-1)*self.f0_bins.unsqueeze(0).unsqueeze(0),2)
                pred_loudness = torch.sum(F.softmax(A_out[:,:,self.n_bins:],-1)*self.loudness_bins.unsqueeze(0).unsqueeze(0),2)
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


"""
##############################################################################
## test

# torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/dataset/dataset.pickle'
ddsp_path = '/Users/adrienbitton/Desktop/intership-expressive-DDSP/results/ddsp_debug_pretrained.ts'

batch_size = 3
train_len = 1000

train_data,test_data,f0_range,loudness_range = load_and_pp_data(file,train_len, articulation_percent = 0.1, test_percent = 0.2)

N_features = len(train_data.keys())
fnames = list(train_data.keys())

model = ModelContinuousPitch_FandA(4, 128, f0_range, loudness_range, n_out=2, n_bins=64, f0_weight=10., ddsp_path=ddsp_path)
model.train()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

mb_input,mb_target = sample_minibatch(train_data,batch_size,train_len,device)

## checking gradient

gradient_check(model,optimizer,mb_input,mb_target)

## checking generation loop and audio synthesis

export_minibatch(model,device,mb_input,mb_target,"./train_",sample_rate=16000)
"""


