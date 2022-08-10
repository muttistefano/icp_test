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
tf_files_2     = []
laser_files_2  = []
tf_files_3     = []
laser_files_3  = []
tf_files_4     = []
laser_files_4  = []
tf_files_5     = []
laser_files_5  = []
tf_files_6     = []
laser_files_6  = []

# main_path_1 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/1/"
# main_path_2 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/2/"
# main_path_3 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/3/"
# main_path_4 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/4/"
# main_path_5 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/5/"
# main_path_6 = "/home/kolmogorov/Documents/ROS/wimpy_destroyer/wimpy_ws_ros2/src/wimpy_acquire/logs/6/"

main_path_1 = "/home/blanker/icp_test/log/pre_train/1/"
main_path_2 = "/home/blanker/icp_test/log/pre_train/2/"
main_path_3 = "/home/blanker/icp_test/log/pre_train/3/"
main_path_4 = "/home/blanker/icp_test/log/pre_train/4/"
main_path_5 = "/home/blanker/icp_test/log/pre_train/5/"
main_path_6 = "/home/blanker/icp_test/log/pre_train/6/"

for root, dirs, files in os.walk(main_path_1):
    for name in files:
        if name.startswith("tf_"):
            tf_files_1.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_1.append(root + "/" + name)

for root, dirs, files in os.walk(main_path_2):
    for name in files:
        if name.startswith("tf_"):
            tf_files_2.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_2.append(root + "/" + name)

for root, dirs, files in os.walk(main_path_3):
    for name in files:
        if name.startswith("tf_"):
            tf_files_3.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_3.append(root + "/" + name)

for root, dirs, files in os.walk(main_path_4):
    for name in files:
        if name.startswith("tf_"):
            tf_files_4.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_4.append(root + "/" + name)

for root, dirs, files in os.walk(main_path_5):
    for name in files:
        if name.startswith("tf_"):
            tf_files_5.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_5.append(root + "/" + name)

for root, dirs, files in os.walk(main_path_6):
    for name in files:
        if name.startswith("tf_"):
            tf_files_6.append(root + "/" + name)
        if name.startswith("ls_"):
            laser_files_6.append(root + "/" + name)


tf_files_1          = natsorted(tf_files_1   ,  key=lambda y: y.lower())
laser_files_1       = natsorted(laser_files_1,  key=lambda y: y.lower())
tf_files_2          = natsorted(tf_files_2   ,  key=lambda y: y.lower())
laser_files_2       = natsorted(laser_files_2,  key=lambda y: y.lower())
tf_files_3          = natsorted(tf_files_3   ,  key=lambda y: y.lower())
laser_files_3       = natsorted(laser_files_3,  key=lambda y: y.lower())
tf_files_4          = natsorted(tf_files_4   ,  key=lambda y: y.lower())
laser_files_4       = natsorted(laser_files_4,  key=lambda y: y.lower())
tf_files_5          = natsorted(tf_files_5   ,  key=lambda y: y.lower())
laser_files_5       = natsorted(laser_files_5,  key=lambda y: y.lower())
tf_files_6          = natsorted(tf_files_6   ,  key=lambda y: y.lower())
laser_files_6       = natsorted(laser_files_6,  key=lambda y: y.lower())

tf_array_1          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_1])
laser_array_1       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_1])
tf_array_2          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_2])
laser_array_2       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_2])
tf_array_3          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_3])
laser_array_3       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_3])
tf_array_4          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_4])
laser_array_4       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_4])
tf_array_5          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_5])
laser_array_5       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_5])
tf_array_6          = np.asarray([tf_to_mat(np.load(file_name)) for file_name in tf_files_6])
laser_array_6       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files_6])



laser_array_1 = laser_array_1[:,:,120:630]
laser_array_2 = laser_array_2[:,:,120:630]
laser_array_3 = laser_array_3[:,:,120:630]
laser_array_4 = laser_array_4[:,:,120:630]
laser_array_5 = laser_array_5[:,:,120:630]
laser_array_6 = laser_array_6[:,:,120:630]

laser_array_1[laser_array_1 == np.inf] = 0
laser_array_2[laser_array_2 == np.inf] = 0
laser_array_3[laser_array_3 == np.inf] = 0
laser_array_4[laser_array_4 == np.inf] = 0
laser_array_5[laser_array_5 == np.inf] = 0
laser_array_6[laser_array_6 == np.inf] = 0

laser_array_1[laser_array_1 > 3] = 0
laser_array_2[laser_array_2 > 3] = 0
laser_array_3[laser_array_3 > 3] = 0
laser_array_4[laser_array_4 > 3] = 0
laser_array_5[laser_array_5 > 3] = 0
laser_array_6[laser_array_6 > 3] = 0


print(tf_array_1.shape)
print(laser_array_1.shape)
print(len(tf_files_1),len(laser_files_1))
print(len(tf_files_2),len(laser_files_2))
print(len(tf_files_3),len(laser_files_3))
print(len(tf_files_4),len(laser_files_4))
print(len(tf_files_5),len(laser_files_5))
print(len(tf_files_6),len(laser_files_6))


laser_tot = []
tf_tot    = []

for las,tf in zip([laser_array_1,laser_array_2,laser_array_3,laser_array_4,laser_array_5,laser_array_6],
                  [tf_array_1,tf_array_2,tf_array_3,tf_array_4,tf_array_5,tf_array_6] ):
    for cnt,(las_s,tf_s) in enumerate(zip(las,tf)):
        rnd_idx_all = np.random.randint(max(0,cnt-20),min(len(las),cnt+20),size=(30))
        rnd_idx = np.setdiff1d(rnd_idx_all,cnt)
        for elem in rnd_idx:
            laser_tot.append(np.concatenate((las_s,las[elem])))
            a_b = np.linalg.inv(tf_s) @ tf[elem]
            tf_tot.append(np.array([a_b[0,2],a_b[1,2],np.arctan2(a_b[1,0],a_b[0,0])]))

laser_tot = np.asarray(laser_tot)
tf_tot    = np.asarray(tf_tot)

print(len(laser_tot))
print(laser_tot.nbytes * 1e-6)
print(tf_tot.nbytes * 1e-6)

# print("lasers mean and std: " + str(laser_tot.mean()) + " " + str(laser_tot.std()))
# print("tf     mean and std: " + str(tf_tot.mean())    + " " + str(tf_tot.std()))
data_std_mean = np.array([laser_tot.mean(),laser_tot.std(),tf_tot.mean(),tf_tot.std()]) 
laser_tot     = (laser_tot - laser_tot.mean())/(laser_tot.std())
tf_tot        = (tf_tot    - tf_tot.mean())/(tf_tot.std())

# tf_min_max = np.array([tf_tot[:,0].min(),tf_tot[:,0].max(),tf_tot[:,1].min(),tf_tot[:,1].max(),tf_tot[:,2].min(),tf_tot[:,2].max()])
# print("tf     min and max: " + str(tf_tot[:,0].min())    + " " + str(tf_tot[:,0].max()))
# print("tf     min and max: " + str(tf_tot[:,1].min())    + " " + str(tf_tot[:,1].max()))
# print("tf     min and max: " + str(tf_tot[:,2].min())    + " " + str(tf_tot[:,2].max()))
# tf_tot[:,0] = (tf_tot[:,0] - tf_tot[:,0].min()) / (tf_tot[:,0].max() - tf_tot[:,0].min())
# tf_tot[:,1] = (tf_tot[:,1] - tf_tot[:,1].min()) / (tf_tot[:,1].max() - tf_tot[:,1].min())
# tf_tot[:,2] = (tf_tot[:,2] - tf_tot[:,2].min()) / (tf_tot[:,2].max() - tf_tot[:,2].min())


# np.save("tf_min_max",tf_min_max)
np.save("data_std_mean",data_std_mean)
np.save("laser",laser_tot)
np.save("tf",tf_tot)



