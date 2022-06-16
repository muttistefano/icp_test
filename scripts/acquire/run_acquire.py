import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import TriggerRequest, Trigger
from icp_docking.srv import move_to_pclRequest,move_to_pcl
import time
import random
import numpy as np
import sys

if __name__ == "__main__":
    rospy.init_node('run_acquire')
    rospy.wait_for_service('/icp_docking/move_to_pcl')
    rospy.wait_for_service('/snap_azure')
    move_to_pcl = rospy.ServiceProxy('/icp_docking/move_to_pcl', move_to_pcl)
    snap_azure  = rospy.ServiceProxy('/snap_azure', Trigger)
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rt = rospy.Rate(50)


    for i in range(1000):

        print("moving")
        start = time.time()
        vel = Twist()
        vel.linear.x  = random.uniform(-0.02, 0.02)
        vel.linear.y  = random.uniform(-0.02, 0.02)
        vel.angular.z = random.uniform(-0.02, 0.02)
        while((time.time() - start) < 3.5) :
            cmd_vel_pub.publish(vel)
            rt.sleep()

        print("docking")
        move_to = move_to_pclRequest()
        move_to.index = 1
        resp = move_to_pcl(move_to)
        print(resp.err_x,resp.err_y,resp.err_z)

        scan_list = []
        
        rospy.sleep(0.5)

        print("snapping")
        sss = snap_azure()
        np.save(sys.argv[1] + "err_"    + sss.message,np.array([resp.err_x,resp.err_y,resp.err_z]))

        rospy.sleep(0.5)

