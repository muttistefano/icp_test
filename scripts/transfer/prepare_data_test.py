from cProfile import label
from dt_apriltags import Detector
import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
from math import sin,cos
from natsort import natsorted


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


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

def tf_to_mat(tf):
    return np.array([[cos(tf[2]),-sin(tf[2]),tf[0]],
                     [sin(tf[2]), cos(tf[2]),tf[1]],
                     [0,0,1]])

tf_files_1     = []
laser_files_1  = []


# main_path_1 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/1/"
# main_path_2 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/2/"
# main_path_3 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/3/"
# main_path_4 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/4/"
# main_path_5 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/5/"
main_path_1 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/test/"

# main_path_1 = "/home/mutti/logs_icp/logs/test/"

for root, dirs, files in os.walk(main_path_1):
    for name in files:
        if name.startswith("tf_"):
            tf_files_1.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_1.append(root + "/" + name)



tf_files_1          = natsorted(tf_files_1   ,  key=lambda y: y.lower())
laser_files_1       = natsorted(laser_files_1,  key=lambda y: y.lower())

tf_array_1          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_1])
laser_array_1       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_1])


laser_array_1 = laser_array_1[:,:,120:630]

laser_array_1[laser_array_1 == np.inf] = 0

laser_array_1[laser_array_1 > 3] = 0


print(tf_array_1.shape)
print(laser_array_1.shape)
print(len(tf_files_1),len(laser_files_1))


laser_tot = []
tf_tot    = []


for cnt,(las_s,tf_s) in enumerate(zip(laser_array_1,tf_array_1)):
    rnd_idx_all = np.random.randint(0,len(laser_array_1),size=(25))
    rnd_idx = np.setdiff1d(rnd_idx_all,cnt)
    for elem in rnd_idx:
        laser_tot.append(np.concatenate((las_s,laser_array_1[elem])))
        a_b = np.linalg.inv(tf_s) @ tf_array_1[elem]
        tf_tot.append(np.array([a_b[0,2],a_b[1,2],np.arctan2(a_b[1,0],a_b[0,0])]))

laser_tot = np.asarray(laser_tot)
tf_tot    = np.asarray(tf_tot)

print(len(laser_tot))
print(laser_tot.nbytes * 1e-6)
print(tf_tot.nbytes * 1e-6)

# print("lasers mean and std: " + str(laser_tot.mean()) + " " + str(laser_tot.std()))
# print("tf     mean and std: " + str(tf_tot.mean())    + " " + str(tf_tot.std()))
# laser_tot = (laser_tot - laser_tot.mean())/(laser_tot.std())
# tf_tot      = (tf_tot - tf_tot.mean())/(tf_tot.std())

print("laser max : " + str(laser_tot.max()))
# laser_tot /= laser_tot.max()

np.save("laser_test",laser_tot)
np.save("tf_test",tf_tot)



