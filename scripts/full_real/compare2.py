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


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")


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

images_files  = []
laser_files   = []
images_files2 = []
laser_files2  = []
error_files   = []

main_path = sys.argv[1]

for file in os.listdir(main_path):
    if file.startswith("color_"):
        images_files.append(main_path + file)
    if file.startswith("laser_"):
        laser_files.append(main_path + file)

images_files          = natsorted(images_files, key=lambda y: y.lower())
laser_files           = natsorted(laser_files,  key=lambda y: y.lower())

main_path2 = sys.argv[2]

for file in os.listdir(main_path2):
    if file.startswith("color_"):
        images_files2.append(main_path2 + file)
    if file.startswith("laser_"):
        laser_files2.append(main_path2 + file)

images_files2          = natsorted(images_files2, key=lambda y: y.lower())
laser_files2           = natsorted(laser_files2,  key=lambda y: y.lower())



print(len(images_files),len(laser_files),len(images_files2),len(laser_files2))

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


if (os.path.isfile(main_path2 + "aptil_tf_1.npy") and os.path.isfile(main_path2 + "aptil_tf_2.npy")):
    print("aptil_tf file found")
    april_tf_1_2         = np.load(main_path2 + "aptil_tf_1.npy")
    april_tf_2_2         = np.load(main_path2 + "aptil_tf_2.npy")
else:
    print("Computingh aptil_tf  file")
    
    april_tf_1_2 = []
    april_tf_2_2 = []

    for cnt,fl in enumerate(images_files2):
        print(cnt," : ",len(images_files2))
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
                april_tf_1_2.append(TT)
            if tag.tag_id==2:
                TT = np.eye(4)
                TT[0:3,0:3] = tag.pose_R
                TT[0,3] = tag.pose_t[0]
                TT[1,3] = tag.pose_t[1]
                TT[2,3] = tag.pose_t[2]
                april_tf_2_2.append(TT)

    april_tf_1_2 = np.asarray(april_tf_1_2)
    april_tf_2_2 = np.asarray(april_tf_2_2)
    np.save(main_path2 + "aptil_tf_1.npy",april_tf_1_2)
    np.save(main_path2 + "aptil_tf_2.npy",april_tf_2_2)

## ref apritags

if (os.path.isfile(main_path2 + "aptil_tf_ref_1.npy") and os.path.isfile(main_path2 + "aptil_tf_ref_2.npy")):
    print("aptil_tf_ref file found")
    aptil_tf_ref_1_2         = np.load(main_path2 + "aptil_tf_ref_1.npy")
    aptil_tf_ref_2_2         = np.load(main_path2 + "aptil_tf_ref_2.npy")
else:
    print("Computingh aptil_tf_ref  file")

    img = cv2.imread(main_path2 + "ref.tiff",cv2.IMREAD_GRAYSCALE)

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
            aptil_tf_ref_1_2 = np.eye(4)
            aptil_tf_ref_1_2[0:3,0:3] = tag.pose_R
            aptil_tf_ref_1_2[0,3] = tag.pose_t[0]
            aptil_tf_ref_1_2[1,3] = tag.pose_t[1]
            aptil_tf_ref_1_2[2,3] = tag.pose_t[2]
        if tag.tag_id==2:
            aptil_tf_ref_2_2 = np.eye(4)
            aptil_tf_ref_2_2[0:3,0:3] = tag.pose_R
            aptil_tf_ref_2_2[0,3] = tag.pose_t[0]
            aptil_tf_ref_2_2[1,3] = tag.pose_t[1]
            aptil_tf_ref_2_2[2,3] = tag.pose_t[2]

    np.save(main_path2 + "aptil_tf_ref_1.npy",aptil_tf_ref_1_2)
    np.save(main_path2 + "aptil_tf_ref_2.npy",aptil_tf_ref_2_2)




fix_m = np.zeros((4,4))
fix_m[0,1] = -1
fix_m[1,0] =  1
fix_m[2,2] =  1
fix_m[3,3] =  1

for cnt in range(len(april_tf_1)):
    april_tf_1[cnt] = fix_m @ april_tf_1[cnt]

for cnt in range(len(april_tf_1_2)):
    april_tf_1_2[cnt] = fix_m @ april_tf_1_2[cnt]

aptil_tf_ref_1   = fix_m @ aptil_tf_ref_1
aptil_tf_ref_1_2 = fix_m @ aptil_tf_ref_1_2

apr_err_1 = []
apr_err_2 = []

apr_err_1_2 = []
apr_err_2_2 = []

for a1,a2 in zip(april_tf_1,april_tf_2):
    apr_err_1.append(np.linalg.inv(a1) @ aptil_tf_ref_1)
    apr_err_2.append(np.linalg.inv(a2) @ aptil_tf_ref_2)

for a1,a2 in zip(april_tf_1_2,april_tf_2_2):
    apr_err_1_2.append(np.linalg.inv(a1) @ aptil_tf_ref_1_2)
    apr_err_2_2.append(np.linalg.inv(a2) @ aptil_tf_ref_2_2)

# print(np.mean([math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3])) for x in apr_err_1]))
# print(np.mean([math.sqrt((x[0,3]*x[0,3]) + (x[1,3]*x[1,3])) for x in apr_err_1_2]))
print("\n")

print("x")
print(np.mean([np.abs(x[0,3]) for x in apr_err_1]))
print(np.mean([np.abs(x[0,3]) for x in apr_err_1_2]))
print("y")
print(np.mean([np.abs(x[1,3]) for x in apr_err_1]))
print(np.mean([np.abs(x[1,3]) for x in apr_err_1_2]))
print("w")
print(np.mean(np.abs([rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_1])))
print(np.mean(np.abs([rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_1_2])))


print("x")
print(np.mean([np.abs(x[0,3]) for x in apr_err_2]))
print(np.mean([np.abs(x[0,3]) for x in apr_err_2_2]))
print("y")
print(np.mean([np.abs(x[1,3]) for x in apr_err_2]))
print(np.mean([np.abs(x[1,3]) for x in apr_err_2_2]))
print("w")
print(np.mean(np.abs([rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_2])))
print(np.mean(np.abs([rotationMatrixToEulerAngles(x[0:3,0:3])[2] for x in apr_err_2_2])))



if True:
    fig2, axs2 = plt.subplots(nrows=3, ncols=1)

    
    axs2[0].plot(range(len(apr_err_1))  ,[np.abs(x[0,3]) for x in apr_err_1],label="apr1")
    axs2[0].plot(range(len(apr_err_1_2)),[np.abs(x[0,3]) for x in apr_err_1_2],label="apr1_2")
    axs2[1].plot(range(len(apr_err_1))  ,[np.abs(x[1,3]) for x in apr_err_1],label="apr1")
    axs2[1].plot(range(len(apr_err_1_2)),[np.abs(x[1,3]) for x in apr_err_1_2],label="apr1_2")
    axs2[2].plot(range(len(apr_err_1))  ,[np.abs(rotationMatrixToEulerAngles(x[0:3,0:3])[2]) for x in apr_err_1],label="apr1")
    axs2[2].plot(range(len(apr_err_1_2)),[np.abs(rotationMatrixToEulerAngles(x[0:3,0:3])[2]) for x in apr_err_1_2],label="apr1_2")
    [x.legend() for x in axs2]
    # plt.show()


if True:
    fig2, axs2 = plt.subplots(nrows=3, ncols=1)

    
    axs2[0].plot(range(len(apr_err_1))  ,[np.abs(x[0,3]) for x in apr_err_2],label="apr1")
    axs2[0].plot(range(len(apr_err_1_2)),[np.abs(x[0,3]) for x in apr_err_2_2],label="apr1_2")
    axs2[1].plot(range(len(apr_err_1))  ,[np.abs(x[1,3]) for x in apr_err_2],label="apr1")
    axs2[1].plot(range(len(apr_err_1_2)),[np.abs(x[1,3]) for x in apr_err_2_2],label="apr1_2")
    axs2[2].plot(range(len(apr_err_1))  ,[np.abs(rotationMatrixToEulerAngles(x[0:3,0:3])[2]) for x in apr_err_2],label="apr1")
    axs2[2].plot(range(len(apr_err_1_2)),[np.abs(rotationMatrixToEulerAngles(x[0:3,0:3])[2]) for x in apr_err_2_2],label="apr1_2")
    [x.legend() for x in axs2]
    plt.show()