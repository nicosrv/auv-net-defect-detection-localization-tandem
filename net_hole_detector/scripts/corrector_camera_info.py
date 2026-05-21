#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo

class CorrectorCameraInfo:

    def __init__(self):
        self.__pub = rospy.Publisher("/stereo/right/camera_info", CameraInfo, queue_size=1)
        self.__sub = rospy.Subscriber("/girona500/xiroi/stereo_ch3/right_optical/camera_info", CameraInfo, self.callback)


    def callback(self, msg):
        message_p =list (msg.P)
        message_p[3] = -73.623  # Tx = -fx * baseline
        msg.P = tuple(message_p)
        self.__pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("fix_right_camera_info")
    CorrectorCameraInfo()
    rospy.spin()

# Comentari de prova!!!!