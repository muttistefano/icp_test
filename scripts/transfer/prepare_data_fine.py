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

def get_xy_lidar_ranges(msggs):
    return np.asarray([msg.range for msg in msggs])



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

april_tf_1 = []
april_tf_2 = []

## images apritags
if (os.path.isfile(main_path + "aptil_tf_1.npy") and os.path.isfile(main_path + "aptil_tf_2.npy")):
    print("aptil_tf file found")
    april_tf_1         = np.load(main_path + "aptil_tf_1.npy")
    april_tf_2         = np.load(main_path + "aptil_tf_2.npy")
else:
    print("Computingh aptil_tf  file")

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

april_tf_1 = np.asarray(april_tf_1)

## laser ranges
if os.path.isfile(main_path + "ranges_data_array.npy"):
    print("ranges_data_array file found")
    ranges_data_array = np.load(main_path + 'ranges_data_array.npy')
else:
    print("ranges_data_array file not found")
    scan_msg_array       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files])
    # ranges_data_array      = [get_xy_lidar(mssg) for mssg in scan_msg_array]
    ranges_data_array      = np.asarray([np.asarray([ms.ranges for ms in mssg]) for mssg in scan_msg_array])
    np.save(main_path + 'ranges_data_array',ranges_data_array)
    # with open(main_path + 'ranges_data_array', 'wb') as fp:
        # pickle.dump(ranges_data_array, fp)


ranges_data_array = ranges_data_array[:,:,718:1228]

ranges_data_array[ranges_data_array == np.inf] = 0

print(ranges_data_array.shape)
print(april_tf_1.shape)

laser_tot = []
tf_tot    = []

for cnt,(las_s,tf_s) in enumerate(zip(ranges_data_array,april_tf_1)):
    rnd_idx_all = np.random.randint(0,len(ranges_data_array),size=(5))
    rnd_idx = np.setdiff1d(rnd_idx_all,cnt)
    for elem in rnd_idx:
        laser_tot.append(np.concatenate((las_s,ranges_data_array[elem])))
        a_b = np.linalg.inv(tf_s) @ april_tf_1[elem]
        tf_tot.append(np.array([a_b[0,2],a_b[1,2],np.arctan2(a_b[1,0],a_b[0,0])]))

laser_tot = np.asarray(laser_tot)
tf_tot    = np.asarray(tf_tot)

print(len(laser_tot))
print(laser_tot.nbytes * 1e-6)
print(tf_tot.nbytes * 1e-6)

# print("lasers mean and std: " + str(laser_tot.mean()) + " " + str(laser_tot.std()))
# print("tf     mean and std: " + str(tf_tot.mean())    + " " + str(tf_tot.std()))

laser_array = (laser_tot - laser_tot.mean())/(laser_tot.std())
# laser_array = (laser_tot - 1.1850791)/(1.1398817)
# tf_tot      = (tf_tot    -0.00555812)/(0.3416164)


np.save("laser_fine",laser_tot)
np.save("tf_fine",tf_tot)



