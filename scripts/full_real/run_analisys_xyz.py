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
from torch import dropout, float32, from_numpy, flatten, no_grad
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler


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

def get_xy_lidar_single(msgg):
    pnt_x = [0 for i in range(len(msgg.ranges))]
    pnt_y = [0 for i in range(len(msgg.ranges))]
    for cnt in range(0,len(msgg.ranges)):
        rng = msgg.ranges[cnt]
        if ((rng < 0.2 ) or (rng > 1.5)):
            continue
        ang = msgg.angle_min + msgg.angle_increment * cnt
        if ((ang<2.5) or (ang>3.5)):
            continue
        pnt_x[cnt] = pnt_x[cnt] + (cos(msgg.angle_min + msgg.angle_increment * cnt) * rng) 
        pnt_y[cnt] = pnt_y[cnt] + (sin(msgg.angle_min + msgg.angle_increment * cnt) * rng)
    pnt_x = np.asarray(pnt_x)
    pnt_y = np.asarray(pnt_y)
    pnt_x = pnt_x[pnt_x != 0]
    pnt_y = pnt_y[pnt_y != 0]
    pnt_z = np.zeros(len(pnt_y))
    return np.asarray([pnt_x,pnt_y,pnt_z])

def get_xy_lidar(lidar_msg):
    pnt_x = [0 for i in range(len(lidar_msg[0].ranges))]
    pnt_y = [0 for i in range(len(lidar_msg[0].ranges))]
    for msg_num,msgg in enumerate(lidar_msg[0:1]):
        for cnt in range(0,len(msgg.ranges)):
            rng = msgg.ranges[cnt]
            if ((rng < 0.2 ) or (rng > 1.5)):
                continue
            ang = msgg.angle_min + msgg.angle_increment * cnt
            if ((ang<2.5) or (ang>3.5)):
                continue
            pnt_x[cnt] = pnt_x[cnt] + (cos(msgg.angle_min + msgg.angle_increment * cnt) * rng)
            pnt_y[cnt] = pnt_y[cnt] + (sin(msgg.angle_min + msgg.angle_increment * cnt) * rng)
    pnt_x = np.asarray(pnt_x)
    pnt_y = np.asarray(pnt_y)
    pnt_x = pnt_x[pnt_x != 0]
    pnt_y = pnt_y[pnt_y != 0]
    pnt_z = np.zeros(len(pnt_y))
    return np.asarray([pnt_x,pnt_y,pnt_z])



camera_matrix = np.array( [ 1961.051025,    0.,  2044.009521, 0.,  1961.474365,   1562.872437, 0., 0., 1.        ]).reshape((3, 3))
dist_coeff    = np.array([0.509122,-2.729715,0.000408,-0.000195,1.573830,0.386061,-2.545589,1.496876])

images_files = []
laser_files  = []
error_files  = []

main_path = sys.argv[1]

for file in os.listdir(main_path):
    if file.startswith("color_"):
        images_files.append(main_path + file)
    if file.startswith("laser_"):
        laser_files.append(main_path + file)

images_files          = natsorted(images_files, key=lambda y: y.lower())
laser_files           = natsorted(laser_files,  key=lambda y: y.lower())


print(len(images_files),len(laser_files))

## images apritags
if (os.path.isfile(main_path + "aptil_tf_1.npy") and os.path.isfile(main_path + "aptil_tf_2.npy")):
    print("aptil_tf file found")
    april_tf_1         = np.load(main_path + "aptil_tf_1.npy")
    april_tf_2         = np.load(main_path + "aptil_tf_2.npy")
else:
    print("Computingh aptil_tf  file")
    
    april_tf_1 = []
    april_tf_2 = []

    for cnt,fl in enumerate(images_files):
        print(cnt," : ",len(images_files))
        img = cv2.imread(fl,cv2.IMREAD_GRAYSCALE)

        rect_img = cv2.undistort(img, camera_matrix, dist_coeff, None)
        
        

        at_detector = Detector(searchpath=['apriltags'],
                            families='tag36h11',
                            nthreads=4,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

        tags = at_detector.detect(rect_img, estimate_tag_pose=True, camera_params=[1961.051025,1961.474365,2044.00952,1562.872437], tag_size=0.0766)
        
        if not tags:
            print(fl + "tags list is  empty")
        if ((len(tags) != 3) or (len(tags) != 2)):
            print(fl + "tags list is  not 3")


        for tag in tags:
            if tag.tag_id==1:
                TT = np.eye(4)
                TT[0:3,0:3] = tag.pose_R
                TT[0,3] = tag.pose_t[0]
                TT[1,3] = tag.pose_t[1]
                TT[2,3] = tag.pose_t[2]
                april_tf_1.append(TT)
            if tag.tag_id==2:
                TT = np.eye(4)
                TT[0:3,0:3] = tag.pose_R
                TT[0,3] = tag.pose_t[0]
                TT[1,3] = tag.pose_t[1]
                TT[2,3] = tag.pose_t[2]
                april_tf_2.append(TT)

    april_tf_1 = np.asarray(april_tf_1)
    april_tf_2 = np.asarray(april_tf_2)
    np.save(main_path + "aptil_tf_1.npy",april_tf_1)
    np.save(main_path + "aptil_tf_2.npy",april_tf_2)

## ref apritags
if (os.path.isfile(main_path + "aptil_tf_ref_1.npy") and os.path.isfile(main_path + "aptil_tf_ref_2.npy")):
    print("aptil_tf_ref file found")
    aptil_tf_ref_1         = np.load(main_path + "aptil_tf_ref_1.npy")
    aptil_tf_ref_2         = np.load(main_path + "aptil_tf_ref_2.npy")
else:
    print("Computingh aptil_tf_ref  file")

    img = cv2.imread(main_path + "ref.tiff",cv2.IMREAD_GRAYSCALE)

    rect_img = cv2.undistort(img, camera_matrix, dist_coeff, None)
    

    at_detector = Detector(searchpath=['apriltags'],
                        families='tag36h11',
                        nthreads=4,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    tags = at_detector.detect(rect_img, estimate_tag_pose=True, camera_params=[1961.051025,1961.474365,2044.00952,1562.872437], tag_size=0.0766)
    
    if not tags:
        print("tags list is  empty")
    if len(tags) != 3:
        print("tags list is  not 3")


    for tag in tags:
        if tag.tag_id==1:
            aptil_tf_ref_1 = np.eye(4)
            aptil_tf_ref_1[0:3,0:3] = tag.pose_R
            aptil_tf_ref_1[0,3] = tag.pose_t[0]
            aptil_tf_ref_1[1,3] = tag.pose_t[1]
            aptil_tf_ref_1[2,3] = tag.pose_t[2]
        if tag.tag_id==2:
            aptil_tf_ref_2 = np.eye(4)
            aptil_tf_ref_2[0:3,0:3] = tag.pose_R
            aptil_tf_ref_2[0,3] = tag.pose_t[0]
            aptil_tf_ref_2[1,3] = tag.pose_t[1]
            aptil_tf_ref_2[2,3] = tag.pose_t[2]

    np.save(main_path + "aptil_tf_ref_1.npy",aptil_tf_ref_1)
    np.save(main_path + "aptil_tf_ref_2.npy",aptil_tf_ref_2)


## laser 
if os.path.isfile(main_path + "scan_data_array"):
    print("scan_data_array file found")
    with open (main_path + 'scan_data_array', 'rb') as fp:
        scan_data_array = pickle.load(fp)
    scan_ref                = get_xy_lidar(np.load(main_path + "ref_laser.npy",allow_pickle=True))
else:
    print("scan_data_array file not found")
    scan_msg_array       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files])
    # scan_data_array      = [get_xy_lidar(mssg) for mssg in scan_msg_array]
    scan_data_array      = [[get_xy_lidar_single(ms) for ms in mssg] for mssg in scan_msg_array]
    scan_ref             = get_xy_lidar(np.load(main_path + "ref_laser.npy",allow_pickle=True))
    with open(main_path + 'scan_data_array', 'wb') as fp:
        pickle.dump(scan_data_array, fp)


## laser ranges
if os.path.isfile(main_path + "ranges_data_array"):
    print("ranges_data_array file found")
    with open (main_path + 'ranges_data_array', 'rb') as fp:
        ranges_data_array = pickle.load(fp)
else:
    print("ranges_data_array file not found")
    scan_msg_array       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files])
    # ranges_data_array      = [get_xy_lidar(mssg) for mssg in scan_msg_array]
    ranges_data_array      = np.asarray([np.asarray([get_xy_lidar_ranges(ms) for ms in mssg]) for mssg in scan_msg_array])
    with open(main_path + 'ranges_data_array', 'wb') as fp:
        pickle.dump(ranges_data_array, fp)

#ICP
if (os.path.isfile(main_path + "err_icp_x.npy") and os.path.isfile(main_path + "err_icp_y.npy") and os.path.isfile(main_path + "err_icp_w.npy")):
    print("icp file found")
    err_icp_x         = np.load(main_path + "err_icp_x.npy")
    err_icp_y         = np.load(main_path + "err_icp_y.npy")
    err_icp_w         = np.load(main_path + "err_icp_w.npy")
else:

    print("icp file not found")
    err_icp_x = [[] for i in range(len(scan_data_array))]
    err_icp_y = [[] for i in range(len(scan_data_array))]
    err_icp_w = [[] for i in range(len(scan_data_array))]

    for ctt,elem in enumerate(scan_data_array):
        for s_e in elem:
            cloud_in = pcl.PointCloud()
            cloud_out = pcl.PointCloud()
            cloud_in.from_array(s_e.astype(np.float32).T)
            cloud_out.from_array(scan_ref.astype(np.float32).T)

            icp = cloud_in.make_IterativeClosestPoint()
            converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out,20)
            err_icp_x[ctt].append(transf[0,3])
            err_icp_y[ctt].append(transf[1,3])
            err_icp_w[ctt].append(rotationMatrixToEulerAngles(transf[0:3,0:3])[2])
    
    err_icp_x = np.asarray(err_icp_x)
    err_icp_y = np.asarray(err_icp_y)
    err_icp_w = np.asarray(err_icp_w)
    np.save(main_path + "err_icp_x",err_icp_x)
    np.save(main_path + "err_icp_y",err_icp_y)
    np.save(main_path + "err_icp_w",err_icp_w)


fix_m = np.zeros((4,4))
fix_m[0,1] = -1
fix_m[1,0] =  1
fix_m[2,2] =  1
fix_m[3,3] =  1

for cnt in range(len(april_tf_1)):
    april_tf_1[cnt] = fix_m @ april_tf_1[cnt]

aptil_tf_ref_1 = fix_m @ aptil_tf_ref_1

apr_err_1 = []
apr_err_2 = []

for a1,a2 in zip(april_tf_1,april_tf_2):
    apr_err_1.append(np.linalg.inv(a1) @ aptil_tf_ref_1)
    apr_err_2.append(np.linalg.inv(a2) @ aptil_tf_ref_2)

# print(np.mean([math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_1]))
# print(np.mean([math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_2]))

if False:
    fig2, axs2 = plt.subplots(nrows=2, ncols=1)

    axs2[1].scatter(range(len(apr_err_1)),[rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_1],label="apr1_w")
    axs2[1].scatter(range(len(apr_err_1)),[rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_2],label="apr2_w")
    # axs2[1].scatter(range(len(err_icp_w)),err_icp_w,label="icpw",linestyle='dashed',alpha=0.25)
    axs2[0].scatter(range(len(apr_err_1)),[math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_1],label="apr")
    axs2[0].scatter(range(len(apr_err_1)),[math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_2],label="apr2")
    # for i in range(10):
        # axs2[0].plot(range(len(err_icp_w)),[math.sqrt((x[i]*x[i]) + (y[i]*y[i])) for x,y in zip(err_icp_x,err_icp_y)],label="icp_err",linestyle='dashed',alpha=0.25)
    
    [x.legend() for x in axs2]

if False:
    fig2, axs2 = plt.subplots(nrows=3, ncols=1)

    axs2[2].plot(range(len(apr_err_1)),[rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_1],label="apr1")
    axs2[2].plot(range(len(apr_err_1)),[rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_2],label="apr2")
    axs2[2].plot(range(len(err_icp_w)),err_icp_w,label="icpw",linestyle='dashed',alpha=0.25)
    # axs2[0].plot(range(len(apr_err_1)),[math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_1],label="apr")
    # axs2[0].plot(range(len(apr_err_1)),[math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3]) + (x[2,3]*x[2,3])) for x in apr_err_2],label="apr2")
    axs2[0].plot(range(len(apr_err_1)),[x[0,3] for x in apr_err_1],label="apr1")
    axs2[0].plot(range(len(apr_err_1)),[x[0,3] for x in apr_err_2],label="apr2")
    axs2[0].plot(range(len(err_icp_x)),err_icp_x,label="apr2",alpha=0.25)
    axs2[1].plot(range(len(err_icp_x)),err_icp_y,label="apr2",alpha=0.25)
    axs2[1].plot(range(len(apr_err_1)),[x[1,3] for x in apr_err_1],label="apr1")
    axs2[1].plot(range(len(apr_err_1)),[x[1,3] for x in apr_err_2],label="apr2")
    # for i in range(10):
        # axs2[0].plot(range(len(err_icp_w)),[math.sqrt((x[i]*x[i]) + (y[i]*y[i])) for x,y in zip(err_icp_x,err_icp_y)],label="icp_err",linestyle='dashed',alpha=0.25)

# plt.show()
# sys.exit()

x_labels = np.asarray( [x[0,3] for x in apr_err_1])
y_labels = np.asarray( [x[1,3] for x in apr_err_1])
w_labels = np.asarray([rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_1])

x_labels = (x_labels - x_labels.mean())/(x_labels.std())
y_labels = (y_labels - y_labels.mean())/(y_labels.std())
w_labels = (w_labels - w_labels.mean())/(w_labels.std())

# labels   = np.asarray([np.asarray([x_labels[cnt],y_labels[cnt],w_labels[cnt]]) for cnt,el in enumerate(w_labels) if np.abs(el) < 0.01])
labels   = np.asarray([np.asarray([x_labels[cnt],y_labels[cnt],w_labels[cnt]]) for cnt,el in enumerate(w_labels)])

print("GPU avail : ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEvice: ",device)



class CustomImageDataset(Dataset):
    def __init__(self, laser_in,tag_label_in, transform_in=None, target_transform_in=None):
        self.laser              = torch.tensor(laser_in,dtype=torch.float32)
        self.tag_label          = torch.tensor(tag_label_in,dtype=torch.float32)
        self.transform          = transform_in
        self.target_transform   = target_transform_in
        self.outputs            = []

    def __len__(self):
        return len(self.tag_label)

    def __getitem__(self, idx):
        # print(idx,self.laser[idx].shape)
        return self.laser[idx], self.tag_label[idx]

np.random.seed(int(time.time()))

def aug(inp_data,lab_data):
    print(inp_data.shape)
    print(lab_data.shape)
    aug_data = []
    aug_lab  = []
    for dt,lb in zip(inp_data,lab_data):
        for i in range(8):
            # rnd_id = np.random.randint(0, high=10, size=3, dtype=int)
            dt_aug = np.add(dt[i:i+3],np.random.normal(0,0.0005,size=(3,dt.shape[1])))
            aug_data.append(dt_aug)
            aug_lab.append(lb)
    
    return np.asarray(aug_data),np.asarray(aug_lab)


ranges_data_array,labels = aug(ranges_data_array,labels)

ranges_data_array = (ranges_data_array)/1.3
# labels          = (labels)/0.025

# w_lab_th = w_labels[np.abs(w_labels)<0.01]

# print(len(w_lab_th))

# print(ranges_data_array.shape,w_labels.shape)
# plt.figure(1)
# plt.plot(range(len(labels)),labels)
# # plt.figure(2)
# # plt.plot(range(len(w_lab_th)),w_lab_th)
# # plt.plot(range(len(ranges_data_array[0].T)),ranges_data_array[0].T)
# plt.show()
# sys.exit()

set_complete = CustomImageDataset(ranges_data_array.astype(np.float32),labels)

train_size = int(len(set_complete) * 0.8)
test_size  = len(set_complete)  - train_size
train_set, test_set = random_split(set_complete, [train_size,test_size ])
print(len(set_complete),len(train_set),len(test_set))

#8
train_loader = DataLoader(train_set, batch_size=128,shuffle=True, num_workers=6,pin_memory=True,persistent_workers=True)
test_loader  = DataLoader(test_set, batch_size=128,shuffle=True, num_workers=2,pin_memory=True,persistent_workers=True)

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

model.to(device)

# criterion = torch.nn.MSELoss(reduction="sum")
criterionx = torch.nn.MSELoss(reduction="mean")
criteriony = torch.nn.MSELoss(reduction="mean")
criterionz = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.AdamW(model.parameters(),lr=0.00005)
# optimizer = torch.optim.SGD(model.parameters(),lr=0.005)
epochs    = 1000
cntw = 0

loss_valid = []

for epoch in range(epochs):

    running_loss       = 0.0
    running_loss_valid = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # outputs = model(inputs)
        ox,oy,oz = model(inputs)
        # loss = criterion(outputs, labels)

        lossx = criterionx(ox[:,0], labels[:,0])
        lossy = criteriony(oy[:,0], labels[:,1])
        lossz = criterionz(oz[:,0], labels[:,2])

        loss = lossx + lossy + lossz

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
            inputs, labels = data[0].to(device), data[1].to(device)
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)

            ox,oy,oz = model(inputs)
            # loss = criterion(outputs, labels)

            lossx = criterionx(ox[:,0], labels[:,0])
            lossy = criteriony(oy[:,0], labels[:,1])
            lossz = criterionz(oz[:,0], labels[:,2])

            loss =  lossx + lossy + lossz

            running_loss_valid += loss.item()
        # for lbb,ott in zip(labels,outputs):
        writer.add_scalars("x", {
            'label_x': labels[0,0].item(),
            'out_x': ox[0].item(),
        }, cntw)
        writer.add_scalars("y", {
            'label_y': labels[0,1].item(),
            'out_y': oy[0].item(),
        }, cntw)
        writer.add_scalars("W", {
            'label_w': labels[0,2].item(),
            'out_w': oz[0].item(),
        }, cntw)
        cntw = cntw + 1

        running_loss_valid = running_loss_valid/float(len(test_loader))
        loss_valid.append(running_loss_valid)
        writer.add_scalars("loss", {
                           'valid': running_loss_valid,
        }, epoch)
        writer.flush()

# np.save("out/gru_l1_adamw_00002_1500_loss.net",np.asarray(loss_valid))
# torch.save(model.state_dict(), "out/gru_l1_adamw_00002_1500.net")

#np.save("out/gru_l1_sdg_00002_1500_loss.net",np.asarray(loss_valid))
torch.save(model.state_dict(), "out/mammtpd_5000.net")
