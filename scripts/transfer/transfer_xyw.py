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
import copy

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")

writer = SummaryWriter('ppew')

laser_array = np.load("laser_fine.npy")
tf_array    = np.load("tf_fine.npy")

print(laser_array.shape)
print("GPU avail : ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("DEvice: ",device)



class CustomDataset(Dataset):
    def __init__(self, laser_in,tf_label_in, transform_in=None, target_transform_in=None):
        # self.laser              = torch.tensor(laser_in,dtype=torch.float32).to(device)
        # self.tf_label           = torch.tensor(tf_label_in,dtype=torch.float32).to(device)
        self.laser              = laser_in
        self.tf_label           = tf_label_in
        self.transform          = transform_in
        self.target_transform   = target_transform_in
        self.outputs            = []

    def __len__(self):
        return len(self.tf_label) - 1

    def __getitem__(self, idx):
        return self.laser[idx], self.tf_label[idx]



set_complete = CustomDataset(laser_array.astype(np.float32),tf_array)


train_size = int(len(set_complete) * 0.65)
valid_size = int(len(set_complete) * 0.20)
test_size  = len(set_complete)  - train_size - valid_size
train_set, valid_set, test_set = random_split(set_complete, [train_size,valid_size,test_size ])


batch_size_train = 32

train_loader = DataLoader(train_set, batch_size=batch_size_train ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
valid_loader = DataLoader(valid_set, batch_size=batch_size_train ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
test_loader  = DataLoader(test_set , batch_size=1 ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)


class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers,fd_n,fd_e):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn   = nn.GRU(input_size=510,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.0)
        self.fcx   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcx1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcx2  = nn.Linear(in_features=fd_e,out_features=1)
        self.fcy   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcy1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcy2  = nn.Linear(in_features=fd_e,out_features=1)
        self.fcw   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcw1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcw2  = nn.Linear(in_features=fd_e,out_features=1)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out_rnn, _ = self.rnn(x,h0)
        outx       = self.fcx(F.relu(out_rnn[:, -1, :]))
        outx       = self.fcx1(F.relu(outx))
        outx       = self.fcx2(F.relu(outx))
        outy       = self.fcy(F.relu(out_rnn[:, -1, :]))
        outy       = self.fcy1(F.relu(outy))
        outy       = self.fcy2(F.relu(outy))
        outw       = self.fcw(F.relu(out_rnn[:, -1, :]))
        outw       = self.fcw1(F.relu(outw))
        outw       = self.fcw2(F.relu(outw))

        return outx,outy,outw


model = RNN(150,4,150,50)

model.load_state_dict(torch.load("model_xyw.net"))
model.eval()

model.float()
model.to(device)

model_old = copy.deepcopy(model)


# for cnt,child in enumerate(model.children()):
#     print(cnt,child)
#     if (cnt in [0,1,4,7]) :
#         print("stooping: ", child)
#         for param in child.parameters():
#             param.requires_grad = False


criterion = torch.nn.MSELoss()


optimizer = torch.optim.AdamW(model.parameters(),lr=0.0000002)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
epochs    = 3000
cntw = 0

loss_valid = []
loss_train = []
test_eval  = []
test_eval_old = []

for epoch in range(epochs):

    running_loss       = 0.0
    running_loss_valid = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device),torch.tensor(data[1],dtype=torch.float32).to(device)
        # inputs, labels = data[0], data[1]
        outputs_x, outputs_y, outputs_w  = model(inputs)
        out  = torch.cat((outputs_x, outputs_y, outputs_w ),dim=1)
        loss = criterion(out,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss = running_loss/float(len(train_loader))
    print(epoch,": ",epochs)
    writer.add_scalars("loss", {
                        'train': running_loss,
    }, epoch)
    loss_train.append(running_loss)


    model.eval()
    with no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device),torch.tensor(data[1],dtype=torch.float32).to(device)
            # inputs, labels = data[0], data[1]
            outputs_x, outputs_y, outputs_w  = model(inputs)
            out  = torch.cat((outputs_x, outputs_y, outputs_w ),dim=1)
            loss = criterion(out,labels)

            running_loss_valid += loss.item()
        writer.add_scalars("x", {
            'label_x': labels[0,0].item(),
            'out_x': outputs_x[0][0].item(),
        }, cntw)
        writer.add_scalars("y", {
            'label_y': labels[0,1].item(),
            'out_y': outputs_y[0][0].item(),
        }, cntw)
        writer.add_scalars("W", {
            'label_w': labels[0,2].item(),
            'out_w': outputs_w[0][0].item(),
        }, cntw)
        cntw = cntw + 1

        running_loss_valid = running_loss_valid/float(len(valid_loader))
        writer.add_scalars("loss", {
                        'valid': running_loss_valid,
        }, epoch)
        loss_valid.append(running_loss_valid)

        writer.flush()


data_std_mean = np.load("data_std_mean_fine.npy")

tf_mean = data_std_mean[2]
tf_std  = data_std_mean[3]


model.eval()
with no_grad():
    for i, data in enumerate(test_loader, 0):
        # inputs, labels = data[0], data[1]
        inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device),torch.tensor(data[1],dtype=torch.float32).to(device)
        outputs_x, outputs_y, outputs_w  = model(inputs)
        writer.add_scalars("x", {
            'test_label_x': (labels[0,0].item() * tf_std) + tf_mean,
            'test_out_x': (outputs_x[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        writer.add_scalars("y", {
            'test_label_y': (labels[0,1].item() * tf_std) + tf_mean,
            'test_out_y': (outputs_y[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        writer.add_scalars("W", {
            'test_label_w': (labels[0,2].item() * tf_std) + tf_mean,
            'test_out_w': (outputs_w[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        cntw = cntw + 1
        writer.flush()
        test_eval.append(np.array([(labels[0,0].item()* tf_std)+ tf_mean,
                         (outputs_x[0][0].item()* tf_std)+ tf_mean,
                         (labels[0,1].item()* tf_std)+ tf_mean,
                         (outputs_y[0][0].item()* tf_std)+ tf_mean,
                         (labels[0,2].item()* tf_std)+ tf_mean,
                         (outputs_w[0][0].item()* tf_std)+ tf_mean]))

model_old.eval()
with no_grad():
    for i, data in enumerate(test_loader, 0):
        # inputs, labels = data[0], data[1]
        inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device),torch.tensor(data[1],dtype=torch.float32).to(device)
        outputs_x, outputs_y, outputs_w  = model_old(inputs)
        writer.add_scalars("x", {
            'test_label_x': (labels[0,0].item() * tf_std) + tf_mean,
            'test_out_x': (outputs_x[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        writer.add_scalars("y", {
            'test_label_y': (labels[0,1].item() * tf_std) + tf_mean,
            'test_out_y': (outputs_y[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        writer.add_scalars("W", {
            'test_label_w': (labels[0,2].item() * tf_std) + tf_mean,
            'test_out_w': (outputs_w[0][0].item() * tf_std) + tf_mean,
        }, cntw)
        cntw = cntw + 1
        writer.flush()
        test_eval_old.append(np.array([(labels[0,0].item()* tf_std)+ tf_mean,
                         (outputs_x[0][0].item()* tf_std)+ tf_mean,
                         (labels[0,1].item()* tf_std)+ tf_mean,
                         (outputs_y[0][0].item()* tf_std)+ tf_mean,
                         (labels[0,2].item()* tf_std)+ tf_mean,
                         (outputs_w[0][0].item()* tf_std)+ tf_mean]))


torch.save(model.state_dict(), "model_xyw_fine.net")
loss_train = np.asarray(loss_train)
loss_valid = np.asarray(loss_valid)
test_eval  = np.asarray(test_eval)

np.save("loss_train_fine_xyw",loss_train)
np.save("loss_valid_fine_xyw",loss_valid)
np.save("test_eval_fine_xyw",test_eval)
np.save("test_eval_fine_old_xyw",test_eval_old)



