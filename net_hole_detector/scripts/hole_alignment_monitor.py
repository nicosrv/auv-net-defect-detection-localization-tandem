#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class HoleAlignmentMonitor:

    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param(
            "~image_topic",
            "/stereo/left/image_rect_color"
        )

        self.mask_topic = rospy.get_param(
            "~mask_topic",
            "/net_hole_detector/stereo_left/hole_mask"
        )

        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            "/girona500/xiroi/stereo_ch3/left_optical/camera_info"
        )

        self.tolerance_u_px = float(rospy.get_param("~tolerance_u_px", 35.0))
        self.tolerance_v_px = float(rospy.get_param("~tolerance_v_px", 45.0))
        self.min_mask_area_px = int(rospy.get_param("~min_mask_area_px", 100))
        self.stable_frames_required = int(rospy.get_param("~stable_frames_required", 5))

        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None

        self.centered_counter = 0

        self.pub_centered = rospy.Publisher(
            "/net_hole_detector/hole_alignment/centered",
            Bool,
            queue_size=1
        )

        self.pub_error_px = rospy.Publisher(
            "/net_hole_detector/hole_alignment/error_px",
            PointStamped,
            queue_size=1
        )

        self.pub_error_norm_u = rospy.Publisher(
            "/net_hole_detector/hole_alignment/error_u_norm",
            Float32,
            queue_size=1
        )

        self.pub_area = rospy.Publisher(
            "/net_hole_detector/hole_alignment/mask_area",
            Float32,
            queue_size=1
        )

        self.pub_debug = rospy.Publisher(
            "/net_hole_detector/hole_alignment/debug",
            Image,
            queue_size=1
        )

        rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_cb,
            queue_size=1
        )

        sub_img = Subscriber(self.image_topic, Image)
        sub_mask = Subscriber(self.mask_topic, Image)

        self.sync = ApproximateTimeSynchronizer(
            [sub_img, sub_mask],
            queue_size=10,
            slop=0.3
        )
        self.sync.registerCallback(self.cb)

        rospy.loginfo("[HoleAlignmentMonitor] Ready.")
        rospy.loginfo("[HoleAlignmentMonitor] image_topic: %s", self.image_topic)
        rospy.loginfo("[HoleAlignmentMonitor] mask_topic: %s", self.mask_topic)
        rospy.loginfo("[HoleAlignmentMonitor] tolerance_u_px: %.1f", self.tolerance_u_px)
        rospy.loginfo("[HoleAlignmentMonitor] tolerance_v_px: %.1f", self.tolerance_v_px)
        rospy.loginfo("[HoleAlignmentMonitor] stable_frames_required: %d", self.stable_frames_required)

    def camera_info_cb(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def get_largest_component(self, mask):
        """
        Nos quedamos con la componente blanca más grande de la máscara.
        Así evitamos ruido suelto.
        """
        mask_u8 = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )

        if num_labels <= 1:
            return None, 0, None

        # label 0 es el fondo. Buscamos la componente más grande real.
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_idx = int(np.argmax(areas)) + 1

        area = int(stats[best_idx, cv2.CC_STAT_AREA])
        centroid = centroids[best_idx]

        component_mask = labels == best_idx

        return component_mask, area, centroid

    def cb(self, img_msg, mask_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr("[HoleAlignmentMonitor] Error convirtiendo imagen/máscara: %s", str(e))
            return

        h, w = img.shape[:2]

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Si todavía no llegó CameraInfo, usamos centro geométrico.
        if self.cx is None or self.cy is None:
            cx = w / 2.0
            cy = h / 2.0
        else:
            cx = float(self.cx)
            cy = float(self.cy)

        component_mask, area, centroid = self.get_largest_component(mask)

        debug = img.copy()

        # Centro de cámara
        cv2.drawMarker(
            debug,
            (int(cx), int(cy)),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=30,
            thickness=2
        )

        centered_msg = Bool()
        centered_msg.data = False

        if component_mask is None or area < self.min_mask_area_px:
            self.centered_counter = 0

            rospy.logwarn_throttle(
                1.0,
                "[HoleAlignmentMonitor] Máscara no válida o demasiado pequeña. area=%d",
                area
            )

            self.pub_centered.publish(centered_msg)
            self.pub_area.publish(Float32(float(area)))

            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = img_msg.header
            self.pub_debug.publish(debug_msg)
            return

        u_hole = float(centroid[0])
        v_hole = float(centroid[1])

        error_u = u_hole - cx
        error_v = v_hole - cy

        area_msg = Float32()
        area_msg.data = float(area)
        self.pub_area.publish(area_msg)

        err_msg = PointStamped()
        err_msg.header = img_msg.header
        err_msg.point.x = error_u
        err_msg.point.y = error_v
        err_msg.point.z = float(area)
        self.pub_error_px.publish(err_msg)

        # Error horizontal normalizado:
        # negativo = agujero a la izquierda
        # positivo = agujero a la derecha
        err_norm = Float32()
        err_norm.data = float(error_u / (w / 2.0))
        self.pub_error_norm_u.publish(err_norm)

        is_centered_now = (
            abs(error_u) <= self.tolerance_u_px and
            abs(error_v) <= self.tolerance_v_px
        )

        if is_centered_now:
            self.centered_counter += 1
        else:
            self.centered_counter = 0

        centered_msg.data = self.centered_counter >= self.stable_frames_required
        self.pub_centered.publish(centered_msg)

        # Debug visual
        overlay = debug.copy()
        overlay[component_mask] = (0, 255, 0)
        debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0)

        # Centro del agujero
        cv2.circle(debug, (int(u_hole), int(v_hole)), 8, (255, 0, 0), -1)

        # Línea centro cámara -> agujero
        cv2.line(
            debug,
            (int(cx), int(cy)),
            (int(u_hole), int(v_hole)),
            (255, 255, 0),
            2
        )

        txt1 = "err_u=%.1f px err_v=%.1f px area=%d" % (error_u, error_v, area)
        txt2 = "CENTERED=%s stable=%d/%d" % (
            str(centered_msg.data),
            self.centered_counter,
            self.stable_frames_required
        )

        cv2.putText(debug, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = img_msg.header
        self.pub_debug.publish(debug_msg)

        rospy.loginfo_throttle(
            0.5,
            "[HoleAlignmentMonitor] error_u=%.1f px | error_v=%.1f px | centered=%s | stable=%d/%d",
            error_u,
            error_v,
            str(centered_msg.data),
            self.centered_counter,
            self.stable_frames_required
        )


if __name__ == "__main__":
    rospy.init_node("hole_alignment_monitor")
    HoleAlignmentMonitor()
    rospy.spin()
