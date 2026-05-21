#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from net_hole_detector.msg import BoundingBoxArray


class StereoBBoxDistanceEstimator:
    """
    Computes hole distance using stereo bbox centers.

    It does NOT control the robot.
    It only estimates and publishes distance for debugging:

        Z = fx * baseline / disparity

    where disparity is computed from the horizontal position of the hole
    in the left and right rectified images.
    """

    def __init__(self):
        # Topics
        self.left_bbox_topic = rospy.get_param(
            "~left_bbox_topic",
            "/net_hole_detector/stereo_left/bounding_boxes"
        )

        self.right_bbox_topic = rospy.get_param(
            "~right_bbox_topic",
            "/net_hole_detector/stereo_right/bounding_boxes"
        )

        self.left_info_topic = rospy.get_param(
            "~left_info_topic",
            "/girona500/xiroi/stereo_ch3/left_optical/camera_info"
        )

        self.right_info_topic = rospy.get_param(
            "~right_info_topic",
            "/stereo/right/camera_info"
        )

        self.out_topic = rospy.get_param(
            "~out_topic",
            "/net_hole_detector/stereo_bbox_distance"
        )

        # Camera parameters
        self.fx = None
        self.width = None
        self.height = None
        self.baseline = None

        # Filters / validation
        self.min_disparity_px = rospy.get_param("~min_disparity_px", 3.0)
        self.max_vertical_error_px = rospy.get_param("~max_vertical_error_px", 80.0)

        # Subscribers
        self.sub_left_info = rospy.Subscriber(
            self.left_info_topic,
            CameraInfo,
            self.left_info_cb,
            queue_size=1
        )

        self.sub_right_info = rospy.Subscriber(
            self.right_info_topic,
            CameraInfo,
            self.right_info_cb,
            queue_size=1
        )

        left_sub = Subscriber(self.left_bbox_topic, BoundingBoxArray)
        right_sub = Subscriber(self.right_bbox_topic, BoundingBoxArray)

        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=10,
            slop=2.0
        )

        self.sync.registerCallback(self.bboxes_cb)

        self.pub_distance = rospy.Publisher(
            self.out_topic,
            Float64,
            queue_size=1
        )

        rospy.loginfo("[StereoBBox] Node ready.")
        rospy.loginfo("[StereoBBox] Left bbox topic: %s", self.left_bbox_topic)
        rospy.loginfo("[StereoBBox] Right bbox topic: %s", self.right_bbox_topic)
        rospy.loginfo("[StereoBBox] Left info topic: %s", self.left_info_topic)
        rospy.loginfo("[StereoBBox] Right info topic: %s", self.right_info_topic)
        rospy.loginfo("[StereoBBox] Output distance topic: %s", self.out_topic)

    def left_info_cb(self, msg):
        self.fx = msg.K[0]
        self.width = msg.width
        self.height = msg.height

        rospy.loginfo_once(
            "[StereoBBox] Left CameraInfo received | fx=%.3f | width=%d | height=%d",
            self.fx,
            self.width,
            self.height
        )

    def right_info_cb(self, msg):
        if self.fx is None:
            return

        tx = msg.P[3]

        if abs(tx) > 1e-9:
            self.baseline = abs(tx / self.fx)
        else:
            self.baseline = rospy.get_param("~baseline", 0.10)

        rospy.loginfo_once(
            "[StereoBBox] Right CameraInfo received | P[3]=%.3f | baseline=%.4f m",
            tx,
            self.baseline
        )

    @staticmethod
    def best_box(msg):
        if not msg.boxes:
            return None
        return max(msg.boxes, key=lambda b: b.score)

    def bboxes_cb(self, left_msg, right_msg):
        if self.fx is None or self.baseline is None or self.width is None or self.height is None:
            rospy.logwarn_throttle(2.0, "[StereoBBox] Waiting for camera info...")
            return

        left_box = self.best_box(left_msg)
        right_box = self.best_box(right_msg)

        if left_box is None or right_box is None:
            rospy.logwarn_throttle(1.0, "[StereoBBox] Missing left or right bbox.")
            return

        # Normalized bbox center -> pixels
        u_left = left_box.x * self.width
        v_left = left_box.y * self.height

        u_right = right_box.x * self.width
        v_right = right_box.y * self.height

        disp_lr = u_left - u_right
        disp_rl = u_right - u_left
        disp_abs = abs(disp_lr)

        vertical_error = abs(v_left - v_right)

        if vertical_error > self.max_vertical_error_px:
            rospy.logwarn(
                "[StereoBBox] Large vertical mismatch | vL=%.1f vR=%.1f | err=%.1f px",
                v_left,
                v_right,
                vertical_error
            )

        if disp_abs < self.min_disparity_px:
            rospy.logwarn(
                "[StereoBBox] Disparity too small | uL=%.1f uR=%.1f | disp_abs=%.2f px",
                u_left,
                u_right,
                disp_abs
            )
            return

        z_abs = (self.fx * self.baseline) / disp_abs

        # Publish debug distance using absolute disparity.
        # For now this is only for testing, not robot control.
        self.pub_distance.publish(Float64(z_abs))

        rospy.loginfo_throttle(
            0.5,
            "[StereoBBox] uL=%.1f uR=%.1f | vL=%.1f vR=%.1f | "
            "disp_LR=%.1f disp_RL=%.1f disp_abs=%.1f | Z_abs=%.3f m | "
            "scoreL=%.2f scoreR=%.2f",
            u_left,
            u_right,
            v_left,
            v_right,
            disp_lr,
            disp_rl,
            disp_abs,
            z_abs,
            left_box.score,
            right_box.score
        )


if __name__ == "__main__":
    rospy.init_node("stereo_bbox_distance_estimator")
    StereoBBoxDistanceEstimator()
    rospy.spin()
