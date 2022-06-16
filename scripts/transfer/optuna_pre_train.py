import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
from math import sin,cos
from natsort import natsorted
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import float32, from_numpy, flatten, no_grad
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import optuna
from optuna.trial import TrialState
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

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
        return self.laser[idx], self.tf_label[idx]



##----------##----------##----------##----------##----------##----------##----------##

class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]

def define_model(trial):
    n_layers  = trial.suggest_int("n_layers", 2, 3)
    n_hidden  = trial.suggest_categorical("n_hidden", [10, 30, 100, 150])
    n_hidden2 = trial.suggest_categorical("n_hidden2", [3,30, 100])
    layers = []


    layers.append(nn.GRU(input_size=310,hidden_size=n_hidden,num_layers=n_layers,batch_first=True))
    layers.append(extractlastcell())
    layers.append(nn.BatchNorm1d(n_hidden))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=n_hidden,out_features=n_hidden2))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=n_hidden2,out_features=3))


    return nn.Sequential(*layers)


def objective(trial):

    epochs    = 150

    model = define_model(trial).to(device)

    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion_names = trial.suggest_categorical('criterion',["MSELoss", "L1Loss"])
    criterion = getattr(nn, criterion_names)()

    # criterion = torch.nn.MSELoss()
    
    set_complete = CustomDataset(laser_array.astype(np.float32),tf_array)


    train_size = int(len(set_complete) * 0.8)
    test_size  = len(set_complete)  - train_size
    train_set, test_set = random_split(set_complete, [train_size,test_size ])
    print(len(set_complete),len(train_set),len(test_set))

    train_loader = DataLoader(train_set, batch_size=2048  ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
    test_loader  = DataLoader(test_set , batch_size=512   ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)



    loss_valid = []

    for epoch in range(epochs):

        running_loss_valid = 0

        model.train()
        for i, data in enumerate(train_loader, 0):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data[0], data[1]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch,": ",epochs)

        model.eval()
        correct = 0
        with no_grad():
            for i, data in enumerate(test_loader, 0):
                # inputs, labels = data[0].to(device), data[1].to(device)
                inputs, labels = data[0], data[1]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss_valid += loss.item()

        running_loss_valid = running_loss_valid/float(len(test_loader))

        trial.report(running_loss_valid, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return running_loss_valid



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=20,
                                reduction_factor=4, min_early_stopping_rate=1))
    study.optimize(objective, n_trials=5000)
    try:
        pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    except Exception as e: 
        print(e)

    finally:

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        text_file = open("optuna_out.txt", "w")
        text_file.write("  Value: " +  str(trial.value) + "\n")
        for key, value in trial.params.items():
            text_file.write("    {}: {} \n".format(key, value))
        text_file.close()
