import numpy as np
import cv2
import sys
import os
import glob

from pyk4a import Config, PyK4A
from datetime import datetime
import rospy
from std_srvs.srv import Trigger,TriggerResponse
from sensor_msgs.msg import LaserScan
import pyrealsense2 as rs


# Principal Point         : 963.501, 528.838
# Focal Length            : 1377.06, 1376.96
# Distortion Model        : Inverse Brown Conrady
# Distortion Coefficients : [0,0,0,0,0]


DEBUG = False

pipeline = rs.pipeline()
config = rs.config()

def snap_realsense(req):

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image_16 = np.asanyarray(color_frame.get_data())
    color_image = (color_image_16/256).astype('uint8')


    now = datetime.now() 
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    cv2.imwrite(sys.argv[1] + "color_" + str(date_time) + ".tiff", color_image)

    print("scan")
    scan_list = []
    for i in range(10):
        ls = rospy.wait_for_message("/scan",LaserScan)
        scan_list.append(ls)

    scan_list = np.asarray( scan_list)

    np.save(sys.argv[1] + "laser_"    + str(date_time),scan_list)


    answ = TriggerResponse()
    answ.success = True
    answ.message = str(date_time)
    return answ

def main_srv():
    rospy.init_node('snap_azure')


    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.y16, 30)

    pipeline.start(config)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # sensor.set_option(rs.option.exposure, 150.000)
    sensor.set_option(rs.option.enable_auto_exposure, True)

    s = rospy.Service('/snap_realsense', Trigger, snap_realsense)
    print("Ready to snap pic from azure.")
    rospy.spin()


if __name__ == "__main__":
    try:

        main_srv()

    finally:

        pipeline.stop()
