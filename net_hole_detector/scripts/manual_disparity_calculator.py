#!/usr/bin/env python3
import rospy
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from net_hole_detector.srv import Trigger, TriggerResponse
from statistics import mean, median
import numpy as np
from  math import pi, tan
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from net_hole_detector.msg import BoundingBox, BoundingBoxArray, Detection3D, Detection3DArray
import cv2

BASELINE = 0.10

class ManualDisparityCalculator:
    def __init__(self):
        rospy.init_node('manual_stereo_depth')
        
        self.left_info_topic = rospy.get_param("~left_camera_info_topic", "/girona500/xiroi/stereo_ch3/left_optical/camera_info")
        self.right_info_topic = rospy.get_param("~right_camera_info_topic", "/girona500/xiroi/stereo_ch3/right_optical/camera_info")
        self.output_topic = rospy.get_param("~output_topic", "/net_hole_detector/stereo_detections_3d")

        # Variables Initialization
        # --- Left Camera ---
        self.__left_fx = 0
        self.__left_image_width = 0
        self.__left_image_height = 0

        # --- Right Camera ---
        self.__right_fx = 0
        self.__right_image_width = 0
        self.__right_image_height = 0

        # Subscribers
        left_bboxes = Subscriber("/net_hole_detector/stereo_left/bounding_boxes", BoundingBoxArray)
        right_bboxes = Subscriber("/net_hole_detector/stereo_right/bounding_boxes", BoundingBoxArray)

        self.sub_left_info = rospy.Subscriber(self.left_info_topic, CameraInfo, self.left_info_callback)
        self.sub_right_info = rospy.Subscriber(self.right_info_topic, CameraInfo, self.right_info_callback)

        # Publishers
        self.pub_detection = rospy.Publisher(self.output_topic, Detection3DArray, queue_size=1)

        # Callback
        self.aprox_subs = ApproximateTimeSynchronizer([left_bboxes, right_bboxes], queue_size=5, slop=0.1)
        self.aprox_subs.registerCallback(self.callback_disparity)

    def left_info_callback(self, msg):
        try: 
            self.__left_fx = msg.K[0]
            self.__left_fy = msg.K[4]
            self.__left_cx = msg.K[2]
            self.__left_cy = msg.K[5]
            self.__left_image_height = msg.height
            self.__left_image_width = msg.width
            self.sub_left_info.unregister() 
            rospy.loginfo(f"Calibración izquierda recibida y guardada. fx = {self.__left_fx}, fy = {self.__left_fy}, cx = {self.__left_cx}, cy = {self.__left_cy}, height = {self.__left_image_height}, width = {self.__left_image_width} ")
        except Exception as e:
            rospy.logerr("Error calib L - {e}")

    def right_info_callback(self, msg):
        try: 
            self.__right_fx = msg.K[0]
            self.__right_image_width = msg.width
            self.__right_image_height = msg.height
            self.sub_right_info.unregister() 
            rospy.loginfo(f"Calibración derecha recibida y guardada. fx = {self.__right_fx}")
        except Exception as e:
            rospy.logerr("Error calib R - {e}")

    def callback_disparity(self, left_msg, right_msg):
        AREA_TOLERANCE = 0.75
        # Sanity check:
        if self.__left_fx == 0 or self.__right_fx == 0 or not left_msg.boxes or not right_msg.boxes:
            return

        # Get best hole in left camera    :
        left_best_box = self.get_best_score_box(left_msg.boxes)
        if not left_best_box:
            return
        
        # Convert to pixels
        left_px_x = left_best_box.x * self.__left_image_width
        left_px_y = left_best_box.y * self.__left_image_height

        # Look for the hole PAIR (in the same height Y)
        right_match_box = self.find_matching_box_in_right(left_px_y, right_msg.boxes)
        if not right_match_box:
            rospy.logwarn("Bbox in left image, but unable to match it with Bbox in right image at the same height")
            return

        right_px_x = right_match_box.x * self.__right_image_width

        # Calculate disparity
        disparity = abs(left_px_x - right_px_x)

        if disparity <= 0:
            return 
            # Avoiding 0 division if the calculus is not good

        # Area filtering to avoid inacurate disparity calculations
        area_left = left_best_box.h * left_best_box.w
        area_right = right_match_box.h * right_match_box.w

        area_ratio = min(area_left, area_right) / max(area_left, area_right)

        if area_ratio < AREA_TOLERANCE:
            rospy.logwarn(f"Rejected for different area ratio: {area_ratio:.2f}")
            return

        # Z = focal length * BaseLine / disparity
        distance = (self.__left_fx * BASELINE) / disparity

        left_pos_x = (left_px_x - self.__left_cx) * distance / self.__left_fx
        left_pos_y = (left_px_y - self.__left_cy) * distance / self.__left_fy

        rospy.loginfo(f"[DEBUG DET] 3D Left Cam Frame: X={left_pos_x:.3f}, Y={left_pos_y:.3f}, Z={distance:.3f}")

        out_msg = Detection3DArray()
        out_msg.header = left_msg.header
        out_msg.detections = []

        det_3d = Detection3D()
        det_3d.class_id = left_best_box.class_id
        det_3d.score = left_best_box.score
        det_3d.x, det_3d.y, det_3d.z = left_pos_x, left_pos_y, distance

        # Real Size (meters) calculation based on box size 
        det_3d.width = (left_best_box.w * self.__left_image_width * distance) / self.__left_fx
        det_3d.height = (left_best_box.h * self.__left_image_height * distance) / self.__left_fy

        out_msg.detections.append(det_3d)

        self.pub_detection.publish(out_msg)
        
    # Con esto quiero filtrar el mejor score del array
    def get_best_score_box(self, boxes):
        """ Returns BBox with higher score """
        best_box = None
        best_score = -1.0
        for bb in boxes:
            if bb.score > best_score:
                best_score = bb.score
                best_box = bb
        return best_box

    def find_matching_box_in_right(self, left_y_pixel, right_boxes):
        """ Returns the corresponding BBox of the right image at the same height as the left image """
        PX_TOLERANCE = 20
        best_match = None
        min_y_diff = float('inf')

        for bb in right_boxes:
            right_px_y = bb.y * self.__right_image_height
            y_diff = abs(left_y_pixel - right_px_y)

            if y_diff < PX_TOLERANCE and y_diff < min_y_diff:
                best_match = bb
                min_y_diff = y_diff

        return best_match 

if __name__ == '__main__':
    try:
        node = ManualDisparityCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
