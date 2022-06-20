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
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import dropout, float32, from_numpy, flatten, no_grad
from torch.autograd import Variable

pptime = datetime.now().strftime("%d_%m_%Y_%H_%M")

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")

writer = SummaryWriter('ppew')

laser_array = np.load("laser.npy")
tf_array    = np.load("tf.npy")


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
        return self.laser[idx][None,:], self.tf_label[idx]



set_complete = CustomDataset(laser_array.astype(np.float32),tf_array)


train_size = int(len(set_complete) * 0.8)
test_size  = len(set_complete)  - train_size
train_set, test_set = random_split(set_complete, [train_size,test_size ])


batch_size_train = 256

train_loader = DataLoader(train_set, batch_size=batch_size_train ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
test_loader  = DataLoader(test_set , batch_size=256             ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(1,50,kernel_size=(5,5),stride=(5,5),padding=(0,0))
        self.bn1  = nn.BatchNorm2d(50)
        self.mx1  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(50,20,kernel_size=(2,2),stride=(2,2),padding=(0,0))
        self.bn2  = nn.BatchNorm2d(20)
        # self.mx2  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc   = nn.Linear(in_features=500,out_features=200)
        self.fc2  = nn.Linear(in_features=200 ,out_features=100)
        self.fc3  = nn.Linear(in_features=100 ,out_features=3)
        # self.fc3  = nn.Linear(in_features=50,out_features=50)
        # self.fc4  = nn.Linear(in_features=50,out_features=3)


    def forward(self,x):
        # print(x.shape)
        out = F.relu(self.cnn1(x))
        # print(out.shape)
        out = self.mx1(out)
        # print(out.shape)
        out = self.bn1(out)
        out = F.relu(self.cnn2(out))
        # print(out.shape)
        # out = self.mx2(out)
        # print(out.shape)
        out = self.bn2(out)
        out = torch.flatten(out,1)
        # out = self.cnn2(out)
        # print(out.shape)
        out = self.fc(F.relu(out))
        out = self.fc2(F.relu(out))
        out = self.fc3(F.relu(out))
        # out    = self.fc3(F.relu(out))
        # out    = self.fc4(F.relu(out))

        return out


model = CNN()

model.float()
model.to(device)

criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()


optimizer = torch.optim.AdamW(model.parameters(),lr=0.000002)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
epochs    = 4000
cntw = 0

loss_valid = []

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

    model.eval()
    with no_grad():
        for i, data in enumerate(test_loader, 0):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss_valid += loss.item()
        # for lbb,ott in zip(labels,outputs):
        writer.add_scalars("x", {
            'label_x': labels[0,0].item(),
            'out_x': outputs[0][0].item(),
        }, cntw)
        writer.add_scalars("y", {
            'label_y': labels[0,1].item(),
            'out_y': outputs[0][1].item(),
        }, cntw)
        writer.add_scalars("W", {
            'label_w': labels[0,2].item(),
            'out_w': outputs[0][2].item(),
        }, cntw)
        cntw = cntw + 1

        running_loss_valid = running_loss_valid/float(len(test_loader))
        loss_valid.append(running_loss_valid)
        writer.add_scalars("loss", {
                        'valid': running_loss_valid,
        }, epoch)
        writer.flush()

# np.save("out/gru_l1_adamw_00002_1500_loss.net",np.asarray(loss_valid))
torch.save(model.state_dict(), "model.net")

