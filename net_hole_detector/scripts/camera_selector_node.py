#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
import numpy as np
from math import sin, cos, sqrt, pi
from net_hole_detector.srv import CameraSelector, CameraSelectorResponse, Trigger, TriggerRequest

FRONTA_CAMERA_ID = 0
GRIPPER_CAMERA_ID = 1
RIGHT_CAMERA_ID = 2
LEFT_CAMERA_ID = 3
STEREO_CAMERA_ID = 4

class CameraSelectorNode:

    def __init__(self):
        rospy.init_node('camera_selector_node', anonymous=True)

        # Parameters (can be set via rosparam)
        self.frontal_image_topic = rospy.get_param("~frontal_original_image", "/girona500/front_camera/camera/image_raw/compressed")
        self.gripper_image_topic = rospy.get_param("~gripper_original_image", "/girona500/bravo/gripper/camera/image_raw/compressed")
        self.right_image_topic = rospy.get_param("~right_original_image", "/girona500/right_camera/camera/image_raw/compressed")
        self.left_image_topic = rospy.get_param("~left_original_image", "/girona500/left_camera/camera/image_raw/compressed")
        self.stereo_left_image_topic = rospy.get_param("~stereo_left_original_image", "/girona500/xiroi/stereo_ch3/left_optical/image_color/compressed")
        self.output_image_topic = rospy.get_param("~image_to_process", "/net_hole_detector/original_image_to_process/image_raw/compressed")

        # Camera Info
        self.frontal_camera_info_topic = rospy.get_param("~frontal_original_camera_info", "/girona500/front_camera/camera/camera_info")
        self.gripper_camera_info_topic = rospy.get_param("~gripper_original_camera_info", "/girona500/bravo/gripper/camera/camera_info")
        self.right_camera_info_topic = rospy.get_param("~right_original_camera_info", "/girona500/right_camera/camera/camera_info")
        self.left_camera_info_topic = rospy.get_param("~left_original_camera_info", "/girona500/left_camera/camera/camera_info")
        self.stereo_left_camera_info_topic = rospy.get_param("~stereo_left_original_camera_info", "/girona500/xiroi/stereo_ch3/left_optical/camera_info")
        self.output_camera_info_topic = rospy.get_param("~camera_info_to_process", "/net_hole_detector/original_camera_info_to_process/camera_info")

        # Subscribers
        self.image_sub = rospy.Subscriber(
            self.frontal_image_topic,
            CompressedImage,
            self.image_callback,
            queue_size=1
        )
 
        self.camera_info_sub = rospy.Subscriber(
            self.frontal_camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )
 
 
        # Publisher
        self.image_pub = rospy.Publisher(
            self.output_image_topic,
            CompressedImage,
            queue_size=10
        )

        self.camera_info_pub = rospy.Publisher(
            self.output_camera_info_topic,
            CameraInfo,
            queue_size=10
        )

        # Initialize services
        camera_topic_selector_service = rospy.Service('net_hole_detector/camera_selector', CameraSelector, self.camera_topic_selector)
        self.__is_gripper_camera_service = rospy.ServiceProxy('net_hole_detector/blurring_activation_srv', Trigger)
        self.__is_not_gripper_camera_service = rospy.ServiceProxy('net_hole_detector/blurring_deactivation_srv', Trigger)
        self.__request_new_camera_info_service = rospy.ServiceProxy('net_hole_detector/update_camera_info_srv', Trigger)
        self.__request_new_camera_info_stereo_node_service = rospy.ServiceProxy('net_hole_detector/update_stereo_node_camera_info_srv', Trigger)

        rospy.loginfo("Image Geolocalization Node Initialized")
        rospy.spin()


    def image_callback(self, msg):
        # Received image published from desired camera published into output image topic
        new_msg = CompressedImage()
        new_msg.data = msg.data
        new_msg.format = msg.format
        new_msg.header = msg.header
        self.image_pub.publish(new_msg)

    def camera_info_callback(self, msg):
        # Received camera info from desired camera published into output camera info topic
        new_msg = CameraInfo()
        new_msg.header = msg.header
        new_msg.height = msg.height
        new_msg.width = msg.width
        new_msg.P = msg.P
        new_msg.K = msg.K
        new_msg.R = msg.R
        new_msg.distortion_model = msg.distortion_model
        new_msg.D = msg.D
        new_msg.binning_x = msg.binning_x
        new_msg.binning_y = msg.binning_y
        self.camera_info_pub.publish(msg)

    """

    """
    def camera_topic_selector(self, req):
        try:
            selected_id = int(req.message)
            rospy.loginfo('Selector received! Id = ' + str(selected_id))
            
            # Selected image and info camera topics
            if selected_id == FRONTA_CAMERA_ID:
                selected_image_topic = self.frontal_image_topic
                selected_camera_info_topic = self.frontal_camera_info_topic
            elif selected_id == GRIPPER_CAMERA_ID:
                selected_image_topic = self.gripper_image_topic
                selected_camera_info_topic = self.gripper_camera_info_topic
            elif selected_id == RIGHT_CAMERA_ID:
                selected_image_topic = self.right_image_topic
                selected_camera_info_topic = self.right_camera_info_topic
            elif selected_id == LEFT_CAMERA_ID:
                selected_image_topic = self.left_image_topic
                selected_camera_info_topic = self.left_camera_info_topic
            elif selected_id == STEREO_CAMERA_ID:
                selected_image_topic = self.stereo_left_image_topic
                selected_camera_info_topic = self.stereo_left_camera_info_topic
            else:
                response = CameraSelectorResponse()
                response.success = False
                response.message = "Wrong camera id"
                service_request = TriggerRequest()
                self.__is_gripper_camera_service(service_request)
                return response 

            service_request = TriggerRequest()
            self.__is_not_gripper_camera_service(service_request)
            self.__request_new_camera_info_service(service_request)
            self.__request_new_camera_info_stereo_node_service(service_request)

            # Unregister old camera images topic and subscribe to new
            self.image_sub.unregister()
            self.image_sub = rospy.Subscriber(
                selected_image_topic,
                CompressedImage,
                self.image_callback,
                queue_size=1
            )

            # Unregister old camera info topic and subscribe to new
            self.camera_info_sub.unregister()
            self.camera_info_sub = rospy.Subscriber(
                selected_camera_info_topic,
                CameraInfo,
                self.camera_info_callback,
                queue_size=1
            )
    
            # Response message
            response = CameraSelectorResponse()
            response.success = True
            response.message = "Camera selected changed"
            return response

        except Exception as e:
            rospy.logerr('Error when receiving message from service = ' + str(e))
            response = CameraSelectorResponse()
            response.success = False
            response.message = "Wrong camera id"
            return response 


if __name__ == "__main__":
    try:
        CameraSelectorNode()
    except rospy.ROSInterruptException:
        pass