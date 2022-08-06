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


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")

writer = SummaryWriter('ppew')

laser_array = np.load("laser_fine.npy")
tf_array    = np.load("tf_fine.npy")

print(laser_array.shape)
print("GPU avail : ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEvice: ",device)



class CustomDataset(Dataset):
    def __init__(self, laser_in,tf_label_in, transform_in=None, target_transform_in=None):
        self.laser              = torch.tensor(laser_in,dtype=torch.float32).to(device)
        self.tf_label           = torch.tensor(tf_label_in,dtype=torch.float32).to(device)
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


batch_size_train = 8192

train_loader = DataLoader(train_set, batch_size=64             ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
valid_loader = DataLoader(valid_set, batch_size=64            ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
test_loader  = DataLoader(test_set , batch_size=64             ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)



class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=510,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.0)
        self.fc   = nn.Linear(in_features=hidden_size,out_features=30)
        self.fc2  = nn.Linear(in_features=30,out_features=3)
        # self.fc3  = nn.Linear(in_features=50,out_features=50)
        # self.fc4  = nn.Linear(in_features=50,out_features=3)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(x.shape)
        out, _ = self.rnn(x,h0)
        # out, _ = self.rnn(x, (h0, c0))
        out    = self.fc(F.relu(out[:, -1, :]))
        out    = self.fc2(F.relu(out))
        # out    = self.fc3(F.relu(out))
        # out    = self.fc4(F.relu(out))

        return out



model = RNN(150,3)
model.load_state_dict(torch.load("model.net"))
model.eval()

model.float()
model.to(device)



for cnt,child in enumerate(model.children()):
    print(cnt,child)
    if (cnt < 2) :
        for param in child.parameters():
            param.requires_grad = False


criterion = torch.nn.MSELoss(reduction="mean")


optimizer = torch.optim.AdamW(model.parameters(),lr=0.00005)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
epochs    = 1500
cntw = 0

loss_valid = []
loss_train = []


for epoch in range(epochs):

    running_loss       = 0.0
    running_loss_valid = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        # inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = data[0], data[1]
        outputs = model(inputs)
        loss = criterion(outputs, labels)

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
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss_valid += loss.item()
        # for lbb,ott in zip(labels,outputs):
        writer.add_scalars("x", {
            'label_x': labels[0,0].item(),
            'out_x': (outputs[0][0].item()*0.3416164) + 0.00555812,
        }, cntw)
        writer.add_scalars("y", {
            'label_y': labels[0,1].item(),
            'out_y': (outputs[0][1].item()*0.3416164) + 0.00555812,
        }, cntw)
        writer.add_scalars("W", {
            'label_w': labels[0,2].item(),
            'out_w': (outputs[0][2].item()*0.3416164) + 0.00555812,
        }, cntw)
        cntw = cntw + 1

        running_loss_valid = running_loss_valid/float(len(test_loader))
        loss_valid.append(running_loss_valid)
        writer.add_scalars("loss", {
                        'valid': running_loss_valid,
        }, epoch)
        writer.flush()
        loss_valid.append(running_loss_valid)

tf_min_max = np.load("tf_min_max_fine.npy")

# model.eval()
# with no_grad():
#     for i, data in enumerate(test_loader, 0):
#         inputs, labels = data[0], data[1]
#         outputs_x, outputs_y, outputs_w  = model(inputs)
#         writer.add_scalars("x", {
#             'test_label_x': (labels[0,0].item() * (tf_min_max[1] - tf_min_max[0])) + tf_min_max[0],
#             'test_out_x': (outputs_x[0][0].item() * (tf_min_max[1] - tf_min_max[0])) + tf_min_max[0],
#         }, cntw)
#         writer.add_scalars("y", {
#             'test_label_y': (labels[0,1].item() * (tf_min_max[3] - tf_min_max[2]))+ tf_min_max[2],
#             'test_out_y': (outputs_y[0][0].item() * (tf_min_max[3] - tf_min_max[2]))+ tf_min_max[2],
#         }, cntw)
#         writer.add_scalars("W", {
#             'test_label_w': (labels[0,2].item() * (tf_min_max[5] - tf_min_max[4]))+ tf_min_max[4],
#             'test_out_w': (outputs_w[0][0].item() * (tf_min_max[5] - tf_min_max[4]))+ tf_min_max[4],
#         }, cntw)
#         cntw = cntw + 1
#         writer.flush()

model.eval()
with no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0], data[1]
        outputs  = model(inputs)
        writer.add_scalars("x", {
            'test_label_x': labels[0,0].item(),
            'test_out_x': outputs[0][0].item(),
        }, cntw)
        writer.add_scalars("y", {
            'test_label_y': labels[0,1].item(),
            'test_out_y': outputs[0][1].item(),
        }, cntw)
        writer.add_scalars("W", {
            'test_label_w': labels[0,2].item(),
            'test_out_w': outputs[0][2].item(),
        }, cntw)
        cntw = cntw + 1
        writer.flush()


torch.save(model.state_dict(), "model_fine.net")
loss_train = np.asarray(loss_train)
loss_valid = np.asarray(loss_valid)

np.save("loss_train_fine",loss_train)
np.save("loss_valid_fine",loss_valid)

