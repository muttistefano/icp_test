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

laser_tot = np.load("laser.npy")
tf_tot    = np.load("tf.npy")

print("lasers mean and std: " + str(laser_tot.mean()) + " " + str(laser_tot.std()))
print("tf     mean and std: " + str(tf_tot.mean())    + " " + str(tf_tot.std()))









