import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
from math import sin,cos
from natsort import natsorted
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import dropout, float32, from_numpy, flatten, no_grad
from torch.autograd import Variable


writer = SummaryWriter('ppew')

laser_array = np.load("laser_fine.npy")
tf_array    = np.load("tf_fine.npy")


print("GPU avail : ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEvice: ",device)



class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size=510,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.0)
        # self.bn  = nn.BatchNorm1d(hidden_size)
        self.rl  = nn.ReLU()
        # self.drp = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features=hidden_size,out_features=30)
        self.fc2 = nn.Linear(in_features=30,out_features=3)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x,h0)
        # out    = self.bn(out[:, -1, :])
        # out    = self.rl(out)
        out    = self.rl(out[:, -1, :])
        out    = self.fc(out)
        out    = self.rl(out)
        out    = self.fc2(out)
        return out


model = RNN(150,3)
model.load_state_dict(torch.load("model_norm.net"))
model.eval()

model.float()
model.to(device)



with no_grad():
    for i, data in enumerate(zip(laser_array,tf_array), 0):
        print(i)
        inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device)[None, :], torch.tensor(data[1],dtype=torch.float32).to(device)
        outputs = model(inputs)


        writer.add_scalars("x", {
            'label_x': labels[0].item(),
            'out_x': (outputs[0][0].item()*0.3416164) + 0.00555812,
        }, i)
        writer.add_scalars("y", {
            'label_y': labels[1].item(),
            'out_y': (outputs[0][1].item()*0.3416164) + 0.00555812,
        }, i)
        writer.add_scalars("W", {
            'label_w': labels[2].item(),
            'out_w': (outputs[0][2].item()*0.3416164) + 0.00555812,
        }, i)

        writer.flush()


