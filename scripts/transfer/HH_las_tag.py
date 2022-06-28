from cProfile import label
from tkinter.tix import Tree
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
from cycpd import affine_registration, rigid_registration, deformable_registration
from scipy.linalg import logm, expm, sqrtm
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

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

def get_xy_lidar(lidar_msg):
    pnt_x = [[] for i in range(len(lidar_msg))]
    pnt_y = [[] for i in range(len(lidar_msg))]
    for msg_num,msgg in enumerate(lidar_msg):
        for cnt in range(0,len(msgg.ranges)):
            rng = msgg.ranges[cnt]
            pnt_x[msg_num].append(cos(msgg.angle_min + msgg.angle_increment * cnt) * rng)
            pnt_y[msg_num].append(sin(msgg.angle_min + msgg.angle_increment * cnt) * rng)
    pnt_x = np.asarray(pnt_x)
    pnt_y = np.asarray(pnt_y)
    filt_idx = np.logical_not(np.amax(np.asarray([np.asarray(np.var(pnt_x,axis=0)),np.asarray(np.var(pnt_y,axis=0))]).T,1)>0.0001)
    return np.asarray([np.asarray(np.mean(pnt_x,axis=0)),np.asarray(np.mean(pnt_y,axis=0))]).T#.T[filt_idx,:]

def log(R):
    # Rotation matrix logarithm
    theta = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1.0)/2.0)
    if theta == 0:
        theta = 0.0001
    return np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))

def invsqrt(mat):
    u,s,v = np.linalg.svd(mat)
    return u.dot(np.diag(1.0/np.sqrt(s))).dot(v)

def calibrate(A, B):
    #transform pairs A_i, B_i
    N = len(A)
    M = np.zeros((3,3))
    for i in range(N):
        Ra, Rb = A[i][0:3, 0:3], B[i][0:3, 0:3]
        M += np.outer(log(Rb), log(Ra))
    Rx = np.dot(invsqrt(np.dot(M.T, M)), M.T)
    # Rx = eul2rot([0,0,-np.pi])
    C = np.zeros((3*N, 3))
    d = np.zeros((3*N, 1))
    for i in range(N):
        Ra,ta = A[i][0:3, 0:3], A[i][0:3, 3]
        Rb,tb = B[i][0:3, 0:3], B[i][0:3, 3]
        C[3*i:3*i+3, :] = np.eye(3) - Ra
        d[3*i:3*i+3, 0] = ta - np.dot(Rx, tb)

    tx = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))
    return Rx, tx.flatten()

def old_calib(tf_icp,tf_cam):

    M_f = np.zeros(shape=(3,3),dtype=np.float64)
    C_f = np.zeros(shape=(3,3),dtype=np.float64)
    D_f = np.zeros(shape=(3,1),dtype=np.float64)

    for idd in range(len(tf_icp)):
        alpha  = logm(tf_icp[idd][0:3,:3])
        beta   = logm(tf_cam[idd][0:3,:3])
        M_f = M_f + np.matmul(beta,alpha.transpose())
    
    M_f[2,2] = 1

    rot_fin = np.matmul(sqrtm(np.linalg.inv(np.matmul(M_f.transpose(),M_f))),M_f.transpose())

    for ixx in range(len(tf_icp)):
        GG  = np.eye(3,dtype=np.float64)
        C_f = np.concatenate((C_f, GG - tf_cam[ixx][0:3,:3]), axis=0)
        bA  = np.array([tf_icp[ixx][0,3],tf_icp[ixx][1,3],tf_icp[ixx][2,3]]).reshape(3,1)
        bB  = np.array([tf_cam[ixx][0,3],tf_cam[ixx][1,3],tf_cam[ixx][2,3]]).reshape(3,1)
        D_f = np.concatenate((D_f, bA - np.matmul(rot_fin,bB)),axis=0)

    sold = np.matmul(np.matmul(np.linalg.pinv(np.matmul(C_f.transpose(),C_f)),C_f.transpose()),D_f)

    HH_mat = np.eye(4,dtype=np.float32)
    HH_mat[0:3,0:3] = rot_fin
    HH_mat[0,3]   = sold[0]
    HH_mat[1,3]   = sold[1]
    HH_mat[2,3]   = sold[2]
    return HH_mat

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(A, B, init_pose=None, max_iterations=2000, tolerance=0.000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

#KINECT
# camera_matrix = np.array( [ 1961.051025,    0.,  2044.009521, 0.,  1961.474365,   1562.872437, 0., 0., 1.        ]).reshape((3, 3))
# dist_coeff    = np.array([0.509122,-2.729715,0.000408,-0.000195,1.573830,0.386061,-2.545589,1.496876])
#RS
# Principal Point         : 963.501, 528.838
# Focal Length            : 1377.06, 1376.96
# Distortion Model        : Inverse Brown Conrady
# Distortion Coefficients : [0,0,0,0,0]

camera_matrix = np.array( [ 1377.06,    0.,  963.501, 0.,  1376.96,   528.838, 0., 0., 1.        ]).reshape((3, 3))
dist_coeff    = np.array([0,0,0,0,0])

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


## images apritags
if os.path.isfile(main_path + "aptil_tf_1.npy"):
    print("aptil_tf file found")
    april_tf_1         = np.load(main_path + "aptil_tf_1.npy")
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

        # tags = at_detector.detect(rect_img, estimate_tag_pose=True, camera_params=[1961.051025,1961.474365,2044.00952,1562.872437], tag_size=0.0766)
        tags = at_detector.detect(rect_img, estimate_tag_pose=True, camera_params=[1377.06, 1376.96,963.501, 528.838], tag_size=0.0766)
        
        if not tags:
            print(fl + "tags list is  empty")
        if ((len(tags) != 3) or (len(tags) != 2)):
            print(fl + "tags list is  not 3")


        for tag in tags:
            if tag.tag_id==0:
                TT = np.eye(4)
                TT[0:3,0:3] = tag.pose_R
                TT[0,3] = tag.pose_t[0]
                TT[1,3] = tag.pose_t[1]
                TT[2,3] = tag.pose_t[2]
                april_tf_1.append(TT)


    april_tf_1 = np.asarray(april_tf_1)

    np.save(main_path + "aptil_tf_1.npy",april_tf_1)


april_tf_1 = np.asarray(april_tf_1)


## laser ranges
if os.path.isfile(main_path + "ranges_data_array.npy"):
    print("ranges_data_array file found")
    ranges_data_array = np.load(main_path + 'ranges_data_array.npy')
else:
    print("ranges_data_array file not found")
    scan_msg_array       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files])
    ranges_data_array      = np.asarray([np.asarray([ms.ranges for ms in mssg]) for mssg in scan_msg_array])
    np.save(main_path + 'ranges_data_array',ranges_data_array)

## laser pcl
if os.path.isfile(main_path + "pcl_data_array.npy"):
    print("pcl_data_array file found")
    pcl_data_array = np.load(main_path + 'pcl_data_array.npy')
else:
    print("pcl_data_array file not found")
    scan_msg_array       = np.asarray([np.load(file_name,allow_pickle=True) for file_name in laser_files])
    # pcl_data_array      = np.asarray([np.asarray([ms.ranges for ms in mssg]) for mssg in scan_msg_array])
    pcl_data_array      = np.asarray([get_xy_lidar(mssg) for mssg in scan_msg_array])
    np.save(main_path + 'pcl_data_array',pcl_data_array)


ranges_data_array = ranges_data_array[:,:,718:1228]
ranges_data_array[ranges_data_array == np.inf] = 0
ranges_data_array[ranges_data_array > 3] = 0


# ranges_data_array = ranges_data_array[:,:,718:1228]



pcl_data_array[pcl_data_array ==  np.inf] = 0
pcl_data_array[pcl_data_array == -np.inf] = 0
# pcl_data_array[pcl_data_array ==  np.nan] = 0
# ranges_data_array[ranges_data_array > 3] = 0

pcl_data_array_fil = []

for cnt0,bigele in enumerate(pcl_data_array):
    del_arr_pcl = []
    for cnt,ele in enumerate(bigele):
        if (ele[0] == 0) and (ele[1] == 0):
            del_arr_pcl.append(cnt)
        if (ele[0] == np.nan) and (ele[1] == np.nan):
            del_arr_pcl.append(cnt)

    pcl_data_array_fil.append(np.delete(pcl_data_array[cnt0], del_arr_pcl,axis=0))

print(ranges_data_array.shape)
print(april_tf_1.shape)


if os.path.isfile(main_path + "rigid_icp_front.npy"):
    print("ICP rigid file found")
    ICP_data_array_front         = np.load(main_path + "rigid_icp_front.npy",allow_pickle=True)
else:
    print("merge rigid ICP")
    ICP_data_array_front = []
    for cnt,elem in enumerate(pcl_data_array_fil[1:]):
        print(str(cnt) + " : " + str(len(pcl_data_array_fil)))
        a = pcl_data_array_fil[cnt][1:]#[418:1528]
        b = elem[1:]#[418:1528]
        reg = rigid_registration(**{'X': a, 'Y': b, 'max_iterations': 10000, 'tolerance': 0.0000000000000001 , 'scale': False})
        try:
            reg.register()
            ICP_data_array_front.append(reg.get_registration_parameters())
        except:
            print("FAILED")
            ICP_data_array_front.append(np.zeros((4,4)))
            pass
        print(a,b)
        # T,d,i = icp(a,b)
        # ICP_data_array_front.append(T)

    ICP_data_array_front = np.asarray(ICP_data_array_front)
    np.save(main_path + "rigid_icp_front.npy",ICP_data_array_front)


tf_s1 = []
for i in range(1,len(april_tf_1)):
    tf_s1.append(np.linalg.inv(april_tf_1[i-1]) @ april_tf_1[i])
 
# tf_s1 = np.asarray(tf_s1)
# print(pcl_data_array_fil[0].shape)
# plt.plot(pcl_data_array_fil[0][:,0],pcl_data_array_fil[0][:,1])
# plt.plot(pcl_data_array_fil[10][:,0],pcl_data_array_fil[10][:,1])
# plt.show()
# sys.exit()

front_icp_tf2 = []
del_el = []

if True:
    for cnt,asd in enumerate(ICP_data_array_front):
        if len(asd) == 4 :
            del_el.append(cnt)
        else:
            # print(asd)
            scale,M2,t = asd
            # if (np.abs(t[0]) > 0.03) or np.abs(t[1]) > 0.03:
            #     del_el.append(cnt)
            #     continue
            # if (np.abs(t[0]) < 0.01) or np.abs(t[1]) < 0.01:
            #     del_el.append(cnt)
            #     continue
            tf_M2 = np.eye(4)
            tf_M2[0:2,0:2] = M2
            tf_M2[0,3]     = t[0]
            tf_M2[1,3]     = t[1]
            # asd = pcl_data_array[cnt][int(len(pcl_data_array[0])/2)]
            asd = pcl_data_array[cnt][0]
            mat = np.eye(4,4)
            mat[0,3] = asd[0]
            mat[1,3] = asd[1]

            asd2 = pcl_data_array[cnt+1][0]
            mat2 = np.eye(4,4)
            mat2[0,3] = asd2[0]
            mat2[1,3] = asd2[1]
            
            mat_end = mat @ np.linalg.inv(tf_M2) @ np.linalg.inv(mat2)
            # mat_end = mat @ tf_M2 @ np.linalg.inv(mat2)

            # if (np.abs(mat_end[0,3]) > 0.3) or (np.abs(mat_end[1,3]) > 0.3):
            #     del_el.append(cnt)
            #     continue
            
            # front_icp_tf2.append(mat_end)
            front_icp_tf2.append(np.linalg.inv(tf_M2))

if False:
    for cnt,T in enumerate(ICP_data_array_front):


        asd = pcl_data_array[cnt][cnt]
        # if (asd[0] == 0) or (asd[1] == 0):
        #     del_el.append(cnt)
        #     continue
        mat = np.eye(4,4)
        mat[0,3] = asd[0]
        mat[1,3] = asd[1]

        asd2 = pcl_data_array[cnt+1][cnt]
        # if (asd2[0] == 0) or (asd2[1] == 0):
        #     del_el.append(cnt)
        #     continue
        mat2 = np.eye(4,4)
        mat2[0,3] = asd2[0]
        mat2[1,3] = asd2[1]

        T_mat = np.eye(4)
        T_mat[0:2,0:2] = T[0:2,0:2]
        T_mat[0,3]     = T[0,2]
        T_mat[1,3]     = T[1,2]

        mat_end = mat @ T_mat @ np.linalg.inv(mat2)
        # mat_end = mat @ tf_M2 @ np.linalg.inv(mat2)

        # if (np.abs(mat_end[0,3]) > 0.3) or (np.abs(mat_end[1,3]) > 0.3):
        #     del_el.append(cnt)
        #     continue
        
        front_icp_tf2.append(mat_end)
        # front_icp_tf2.append(np.linalg.inv(T_mat))


    front_icp_tf2 = np.asarray(front_icp_tf2)




tf_s1 = np.delete(tf_s1, del_el,axis=0)

# plt.plot(front_icp_tf2[:,0,3],'--',label="icp_x")
# plt.plot(tf_s1[:,0,3],label="tf_x")
# plt.plot(front_icp_tf2[:,1,3],'--',label="icp_y")
# plt.plot(tf_s1[:,1,3],label="tf_y")
# plt.legend()
# plt.show()
# sys.exit()



HH = np.array([[ 1.         , 0.       ,   0.        ,  -0.00        ],
               [ 0.        ,  1.       ,   0.      ,    0.0          ],
               [-0.       ,   0.        ,  1.     ,     0.           ],
               [ 0.         , 0.         , 0.    ,      1.           ]])

if True:

    front_icp_tf2_hh = []
    for el in front_icp_tf2 :
        front_icp_tf2_hh.append(HH @ el @ np.linalg.inv(HH))

    front_icp_tf2_hh = np.asarray(front_icp_tf2_hh)
    
    e_tf_1 = [rot2eul(tf) for tf in front_icp_tf2_hh]
    e_tf_i = [rot2eul(tf) for tf in tf_s1]
    e_tf_1 = np.asarray(e_tf_1)
    e_tf_i = np.asarray(e_tf_i)


    plt.figure("x")
    plt.plot(front_icp_tf2_hh[:,0,3])
    plt.plot(tf_s1[:,0,3])

    plt.figure("y")
    plt.plot(front_icp_tf2_hh[:,1,3])
    plt.plot(tf_s1[:,1,3])

    plt.figure("w")
    plt.plot(e_tf_1[:,2])
    plt.plot(e_tf_i[:,2])
    plt.show()
    # sys.exit()

# A, B = [], []
# for i in range(1,len(front_icp_tf2)):
#     p = tf_s1[i-1], front_icp_tf2[i-1]
#     n = tf_s1[i]  , front_icp_tf2[i]
#     A.append(np.dot(np.linalg.inv(p[0]), n[0]))
#     B.append(np.dot(p[1], np.linalg.inv(n[1])))
#     # A.append(np.dot(p[0], np.linalg.inv(n[0])))
#     # B.append(np.dot(np.linalg.inv(p[1]), n[1]))

# HH_mat = np.eye(4)
# Rx, tx = calibrate(A, B)
# HH_mat[0:3, 0:3] = Rx
# HH_mat[0:3, -1] = tx
# print("HH_MAT")
# print(repr(HH_mat))




# print(old_calib(tf_s1,front_icp_tf2))

