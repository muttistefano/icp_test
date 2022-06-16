import numpy as np
import cv2
import sys
import os
import glob
import pyk4a
from pyk4a import Config, PyK4A
from datetime import datetime
import rospy
from std_srvs.srv import Trigger,TriggerResponse
from sensor_msgs.msg import LaserScan

DEBUG = False


k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_3072P ,
        color_format=pyk4a.ImageFormat.COLOR_BGRA32,
        depth_mode=pyk4a.DepthMode.OFF,
        synchronized_images_only=False,
        camera_fps=pyk4a.FPS.FPS_15,
    )
)


def snap_azure(req):
    capture = k4a.get_capture()
    image = capture.color[:, :, :3].astype(np.uint8)
    now = datetime.now() 
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    cv2.imwrite(sys.argv[1] + "color_" + str(date_time) + ".tiff", image)

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
    k4a.start()
    s = rospy.Service('/snap_azure', Trigger, snap_azure)
    print("Ready to snap pic from azure.")
    rospy.spin()


if __name__ == "__main__":
    try:

        main_srv()

    finally:

        k4a.stop()
