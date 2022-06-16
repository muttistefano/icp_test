from cProfile import label
from dt_apriltags import Detector
import cv2
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
from math import sin,cos
from natsort import natsorted
from sensor_msgs.msg import LaserScan
#import pcl
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
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")

writer = SummaryWriter('ppew')

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    if n < 1e-6:
        return True
    else:
        print(n)
        return True

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def get_xy_lidar_ranges(msgg):
    rng_list = []
    for cnt in range(0,len(msgg.ranges)):
        rng = msgg.ranges[cnt]
        ang = msgg.angle_min + msgg.angle_increment * cnt
        if ((ang<2.5) or (ang>3.5)):
            continue
        if ((rng < 0.2 ) or (rng > 1.5)):
            rng_list.append(0)    
            continue
        rng_list.append(rng)
    return np.asarray(rng_list)


rospy.init_node('rnn_pub', anonymous=True)
err_pub = rospy.Publisher("/err",Float32MultiArray,queue_size=1)

cnt_las = 0
err_w = 0
inp = np.zeros((3,310))

print("GPU avail : ",torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("DEvice: ",device)

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size=310,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.0)
        self.bn  = nn.BatchNorm1d(hidden_size)
        self.rl  = nn.ReLU()
        # self.drp = nn.Dropout(0.1)
        self.fcx = nn.Linear(in_features=hidden_size,out_features=1)
        self.fcy = nn.Linear(in_features=hidden_size,out_features=1)
        self.fcz = nn.Linear(in_features=hidden_size,out_features=1)
        

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(x.shape)
        out, _ = self.lstm(x,h0)
        out    = self.bn(out[:, -1, :])
        out    = self.rl(out)
        # out    = self.drp(out)
        outz    = self.fcz(out)
        # out_px  = torch.cat((outz,out),1)
        outx    = self.fcx(out)
        outy    = self.fcy(out)
        
        # return torch.cat((outx,outy,outz),1)
        return outx,outy,outz



model = RNN(20,3)

model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

def las_call(data):
    global inp, cnt_las,err_w,err_pub
    rng_list = []
    for cnt in range(0,len(data.ranges)):
        rng = data.ranges[cnt]
        ang = data.angle_min + data.angle_increment * cnt
        if ((ang<2.5) or (ang>3.5)):
            continue
        if ((rng < 0.2 ) or (rng > 1.5)):
            rng_list.append(0)    
            continue
        rng_list.append(rng)
    # inp[cnt_las%10] = np.asarray(rng_list)
    inp = np.roll(inp, -1, axis=0)
    inp[0] = (np.asarray(rng_list))/1.3
    cnt_las = cnt_las +1

    inputs = from_numpy(inp.astype(np.float32)).to(device)[None, :]
    outputs = model(inputs)
    # -0.0023598537767612077 0.00861956453476106
    # 0.0009632259974502805 0.008281994060063082
    # -0.004206299619544099 0.0084534111496911
    xe = (outputs[0].detach().cpu().numpy()[0][0]*0.00861956453476106) -0.0023598537767612077
    ye = (outputs[1].detach().cpu().numpy()[0][0]*0.0082819940600630826) +0.0009632259974502805
    we = (outputs[2].detach().cpu().numpy()[0][0]*0.0084534111496911) -0.004206299619544099
    err_msg = Float32MultiArray()
    err_msg.data.append(xe)
    err_msg.data.append(ye)
    err_msg.data.append(we)
    # err_msg.data[0] = xe
    # err_msg.data[1] = ye
    # err_msg.data[2] = we
    err_pub.publish(err_msg)


rospy.Subscriber("/scan", LaserScan, las_call)


rospy.spin()
