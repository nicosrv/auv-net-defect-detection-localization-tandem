#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from net_hole_detector.srv import Trigger, TriggerResponse
import numpy as np
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from net_hole_detector.msg import BoundingBoxArray, Detection3D, Detection3DArray
import cv2


HYSTERESI = 50

# ==========================================================
# FASTSAM + STEREO DEPTH PARAMETERS
# ==========================================================
DEFAULT_MIN_VALID_MASK_PIXELS = 1
DEFAULT_Z_MIN_VALID = 0.05
DEFAULT_Z_MAX_VALID = 10.00
DEFAULT_MAX_Z_SPREAD = 1.50
DEFAULT_MAX_Z_JUMP = 2.00
DEFAULT_TEMPORAL_WINDOW = 5
DEFAULT_MIN_BBOX_SCORE = 0.20
DEFAULT_MAX_JUMP_REJECTS_BEFORE_RESET = 3
DEFAULT_HISTORY_TIMEOUT = 5.0

# Dilation fallback:
# first: exact FastSAM mask
# second: slightly dilated FastSAM mask
DEFAULT_USE_DILATED_MASK_FALLBACK = True
DEFAULT_DILATION_KERNEL_SIZE = 9
DEFAULT_DILATION_ITERATIONS = 3


class StereoDistanceEstimator:

    def __init__(self):

        self.info_topic = rospy.get_param(
            "~camera_info_topic",
            "/girona500/xiroi/stereo_ch3/left_optical/camera_info"
        )

        # Camera intrinsics
        self.__fx = 0.0
        self.__fy = 0.0
        self.__cx = 0.0
        self.__cy = 0.0

        self.bridge = CvBridge()

        # Robust depth parameters
        self.min_valid_mask_pixels = rospy.get_param(
            "~min_valid_mask_pixels",
            DEFAULT_MIN_VALID_MASK_PIXELS
        )

        self.z_min_valid = rospy.get_param(
            "~z_min_valid",
            DEFAULT_Z_MIN_VALID
        )

        self.z_max_valid = rospy.get_param(
            "~z_max_valid",
            DEFAULT_Z_MAX_VALID
        )

        self.max_z_spread = rospy.get_param(
            "~max_z_spread",
            DEFAULT_MAX_Z_SPREAD
        )

        self.max_z_jump = rospy.get_param(
            "~max_z_jump",
            DEFAULT_MAX_Z_JUMP
        )

        self.temporal_window = rospy.get_param(
            "~temporal_window",
            DEFAULT_TEMPORAL_WINDOW
        )

        self.min_bbox_score = rospy.get_param(
            "~min_bbox_score",
            DEFAULT_MIN_BBOX_SCORE
        )

        self.max_jump_rejects_before_reset = rospy.get_param(
            "~max_jump_rejects_before_reset",
            DEFAULT_MAX_JUMP_REJECTS_BEFORE_RESET
        )

        self.history_timeout = rospy.get_param(
            "~history_timeout",
            DEFAULT_HISTORY_TIMEOUT
        )

        self.use_dilated_mask_fallback = rospy.get_param(
            "~use_dilated_mask_fallback",
            DEFAULT_USE_DILATED_MASK_FALLBACK
        )

        self.dilation_kernel_size = rospy.get_param(
            "~dilation_kernel_size",
            DEFAULT_DILATION_KERNEL_SIZE
        )

        self.dilation_iterations = rospy.get_param(
            "~dilation_iterations",
            DEFAULT_DILATION_ITERATIONS
        )

        # Force odd kernel size
        if self.dilation_kernel_size < 3:
            self.dilation_kernel_size = 3

        if self.dilation_kernel_size % 2 == 0:
            self.dilation_kernel_size += 1

        # Temporal history of valid Z measurements
        self.z_history = []
        self.z_jump_rejects = 0
        self.last_valid_z_time = rospy.Time(0)

        # Subscribers
        rospy.Subscriber("/stereo/disparity", DisparityImage, self.callback_disparity)

        sub_points = Subscriber("/stereo/points2", PointCloud2)
        sub_boxes = Subscriber("/net_hole_detector/stereo_left/bounding_boxes", BoundingBoxArray)
        sub_mask = Subscriber("/net_hole_detector/stereo_left/hole_mask", Image)

        self.aprox_subs = ApproximateTimeSynchronizer(
            [sub_points, sub_boxes, sub_mask],
            queue_size=10,
            slop=0.3
        )
        self.aprox_subs.registerCallback(self.callback_distance)

        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)

        # Publishers
        self.pub_disp = rospy.Publisher("/stereo/disparity_image", Image, queue_size=1)
        self.pub_deb = rospy.Publisher("/stereo/deb_image", Image, queue_size=1)

        self.pub_stereo_detect3d = rospy.Publisher(
            "/net_hole_detector/stereo_detections_3d",
            Detection3DArray,
            queue_size=1
        )

        self.update_camera_info_srv = rospy.ServiceProxy(
            "net_hole_detector/update_stereo_node_camera_info_srv",
            Trigger
        )

        rospy.loginfo("[StereoDistanceEstimator] Ready.")
        rospy.loginfo("[StereoDistanceEstimator] Listening points: /stereo/points2")
        rospy.loginfo("[StereoDistanceEstimator] Listening bboxes: /net_hole_detector/stereo_left/bounding_boxes")
        rospy.loginfo("[StereoDistanceEstimator] Listening mask: /net_hole_detector/stereo_left/hole_mask")

        rospy.loginfo(
            "[StereoDistanceEstimator] FastSAM mask depth params | "
            "min_pixels=%d | z_range=[%.2f, %.2f] | max_spread=%.2f | "
            "max_jump=%.2f | temporal_window=%d | min_score=%.2f | "
            "reset_after_rejects=%d | history_timeout=%.1fs | "
            "dilated_fallback=%s | kernel=%d | iterations=%d",
            self.min_valid_mask_pixels,
            self.z_min_valid,
            self.z_max_valid,
            self.max_z_spread,
            self.max_z_jump,
            self.temporal_window,
            self.min_bbox_score,
            self.max_jump_rejects_before_reset,
            self.history_timeout,
            str(self.use_dilated_mask_fallback),
            self.dilation_kernel_size,
            self.dilation_iterations
        )

    def __camera_change_callback(self, req):
        self.sub_info.unregister()
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")

        response = TriggerResponse()
        response.success = True
        response.message = "Request processed!"
        return response

    def info_callback(self, msg):
        """
        Receives camera intrinsics.
        """
        try:
            self.__fx = msg.K[0]
            self.__fy = msg.K[4]
            self.__cx = msg.K[2]
            self.__cy = msg.K[5]

            self.sub_info.unregister()

            rospy.loginfo(
                "Calibración recibida y guardada: fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                self.__fx,
                self.__fy,
                self.__cx,
                self.__cy
            )

        except Exception as e:
            rospy.logerr(f"Error leyendo CameraInfo: {e}")

    def callback_disparity(self, msg):
        """
        Republishes disparity image for visualization.
        """
        self.pub_disp.publish(msg.image)

    def _reset_temporal_filter(self, new_z=None):
        """
        Resets temporal Z history.
        """
        self.z_history = []
        self.z_jump_rejects = 0

        if new_z is not None:
            self.z_history.append(float(new_z))
            self.last_valid_z_time = rospy.Time.now()

    def _update_temporal_filter(self, z_raw):
        """
        Updates temporal median filter.
        """

        z_raw = float(z_raw)
        now = rospy.Time.now()

        # If history is too old, reset it.
        if len(self.z_history) > 0:
            time_since_valid = (now - self.last_valid_z_time).to_sec()

            if time_since_valid > self.history_timeout:
                rospy.logwarn(
                    "[FastSAM DEPTH] Reset del filtro por timeout | "
                    "ultimo dato valido hace %.2fs | nueva Z=%.3fm",
                    time_since_valid,
                    z_raw
                )
                self._reset_temporal_filter(z_raw)
                return z_raw

        # Jump rejection against current temporal median
        if len(self.z_history) > 0:
            z_ref = float(np.median(self.z_history))
            z_jump = abs(z_raw - z_ref)

            if z_jump > self.max_z_jump:
                self.z_jump_rejects += 1

                rospy.logwarn(
                    "[FastSAM DEPTH] Z descartada por salto temporal | "
                    "Z_raw=%.3fm | Z_ref=%.3fm | salto=%.3fm | rechazos=%d/%d",
                    z_raw,
                    z_ref,
                    z_jump,
                    self.z_jump_rejects,
                    self.max_jump_rejects_before_reset
                )

                if self.z_jump_rejects >= self.max_jump_rejects_before_reset:
                    rospy.logwarn(
                        "[FastSAM DEPTH] Reset del filtro temporal. "
                        "Nueva referencia aceptada: %.3fm",
                        z_raw
                    )
                    self._reset_temporal_filter(z_raw)
                    return z_raw

                return None

        # Valid temporal update
        self.z_jump_rejects = 0
        self.z_history.append(z_raw)
        self.last_valid_z_time = now

        if len(self.z_history) > self.temporal_window:
            self.z_history.pop(0)

        return float(np.median(self.z_history))

    def _get_valid_depths_from_mask(self, z_array, candidate_mask):
        """
        Returns valid Z values inside a candidate mask.
        """

        valid_mask = (
            candidate_mask &
            (~np.isnan(z_array)) &
            (z_array > self.z_min_valid) &
            (z_array < self.z_max_valid)
        )

        valid_depths = z_array[valid_mask]

        return valid_depths, valid_mask

    def _make_dilated_mask(self, mask_roi):
        """
        Dilates FastSAM mask slightly to collect depth points around
        the segmented hole contour.
        """

        mask_u8 = (mask_roi.astype(np.uint8)) * 255

        kernel = np.ones(
            (self.dilation_kernel_size, self.dilation_kernel_size),
            dtype=np.uint8
        )

        dilated = cv2.dilate(
            mask_u8,
            kernel,
            iterations=self.dilation_iterations
        )

        return dilated > 0

    def callback_distance(self, msg_point2, msg_bb, msg_mask):
        """
        Receives:
          - /stereo/points2
          - /net_hole_detector/stereo_left/bounding_boxes
          - /net_hole_detector/stereo_left/hole_mask

        Calculates 3D detections using:
          1. exact FastSAM mask
          2. if needed, slightly dilated FastSAM mask
        """

        height = msg_point2.height
        width = msg_point2.width

        # ---------------- READ FASTSAM MASK ----------------
        try:
            mask_img = self.bridge.imgmsg_to_cv2(msg_mask, desired_encoding="mono8")

            if mask_img.shape[0] != height or mask_img.shape[1] != width:
                mask_img = cv2.resize(
                    mask_img,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

        except Exception as e:
            rospy.logwarn(f"[FastSAM MASK] Error leyendo máscara: {e}")
            mask_img = np.zeros((height, width), dtype=np.uint8)

        # ---------------- READ POINT CLOUD ----------------
        try:
            points = list(
                pc2.read_points(
                    msg_point2,
                    field_names=("x", "y", "z"),
                    skip_nans=False
                )
            )
            point_cloud = np.array(points).reshape((height, width, 3))

        except Exception as e:
            rospy.logerr(f"[StereoDistanceEstimator] Error leyendo /stereo/points2: {e}")
            return

        # Debug image
        new_point_cloud = np.zeros((height, width), dtype=np.uint8)

        # ---------------- CASE WITH NO DETECTIONS ----------------
        if not msg_bb.boxes:
            rospy.loginfo_throttle(2.0, "Without detection!")

            h_start = max(0, height // 2 - HYSTERESI)
            h_end = min(height, height // 2 + HYSTERESI)
            w_start = max(0, width // 2 - HYSTERESI)
            w_end = min(width, width // 2 + HYSTERESI)

            central_zone_z = point_cloud[h_start:h_end, w_start:w_end, 2]

            valid_mask = (
                (~np.isnan(central_zone_z)) &
                (central_zone_z > self.z_min_valid) &
                (central_zone_z < self.z_max_valid)
            )

            valid_depths = central_zone_z[valid_mask]

            if valid_depths.size > 0:
                distance = np.median(valid_depths)
                z_min = np.min(valid_depths)
                z_max = np.max(valid_depths)

                rospy.loginfo_throttle(
                    2.0,
                    "[DEBUG NO-DET] Centro Imagen | Z-Median: %.3fm "
                    "(Min: %.3f, Max: %.3f) | Pixels: %d",
                    distance,
                    z_min,
                    z_max,
                    len(valid_depths)
                )
            else:
                rospy.logwarn_throttle(
                    2.0,
                    "[DEBUG NO-DET] Centro Imagen sin datos válidos"
                )

            new_point_cloud[h_start:h_end, w_start:w_end] = 255

            image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding="mono8")
            image_msg.header = msg_point2.header
            self.pub_deb.publish(image_msg)

            return

        # ---------------- CASE WITH DETECTIONS ----------------
        rospy.loginfo_throttle(1.0, "Detection!")

        out_msg = Detection3DArray()
        out_msg.header = msg_point2.header
        out_msg.detections = []

        for bb in msg_bb.boxes:

            if bb.score < self.min_bbox_score:
                rospy.logwarn(
                    "[DEBUG DET] BBox descartada por score bajo: %.2f < %.2f",
                    bb.score,
                    self.min_bbox_score
                )
                continue

            # Convert normalized bbox to pixel coordinates
            u1 = int((bb.x - bb.w / 2.0) * width)
            u2 = int((bb.x + bb.w / 2.0) * width)
            v1 = int((bb.y - bb.h / 2.0) * height)
            v2 = int((bb.y + bb.h / 2.0) * height)

            # Boundaries
            u1 = max(0, min(width - 1, u1))
            u2 = max(0, min(width, u2))
            v1 = max(0, min(height - 1, v1))
            v2 = max(0, min(height, v2))

            if u2 <= u1 or v2 <= v1:
                rospy.logwarn("[DEBUG DET] BBox inválida. Se ignora.")
                continue

            # Draw bbox borders in debug image
            new_point_cloud[v1:v2, u1] = 255
            new_point_cloud[v1:v2, max(u1, u2 - 1)] = 255
            new_point_cloud[v1, u1:u2] = 255
            new_point_cloud[max(v1, v2 - 1), u1:u2] = 255

            # Extract depth inside bbox
            z_array = point_cloud[v1:v2, u1:u2, 2]

            # Extract exact FastSAM mask inside bbox
            exact_mask_roi = mask_img[v1:v2, u1:u2] > 0

            # ------------------------------------------------------
            # STEP 1: exact FastSAM mask
            # ------------------------------------------------------
            valid_depths, valid_mask = self._get_valid_depths_from_mask(
                z_array,
                exact_mask_roi
            )

            used_mask_type = "FastSAM exact mask"
            candidate_mask_for_debug = exact_mask_roi

            # ------------------------------------------------------
            # STEP 2: dilated FastSAM mask fallback
            # ------------------------------------------------------
            if valid_depths.size < self.min_valid_mask_pixels and self.use_dilated_mask_fallback:

                rospy.logwarn(
                    "[FastSAM DEPTH] Pocos puntos en máscara exacta (%d/%d). "
                    "Probando máscara FastSAM dilatada...",
                    valid_depths.size,
                    self.min_valid_mask_pixels
                )

                dilated_mask_roi = self._make_dilated_mask(exact_mask_roi)

                valid_depths_dilated, valid_mask_dilated = self._get_valid_depths_from_mask(
                    z_array,
                    dilated_mask_roi
                )

                if valid_depths_dilated.size >= self.min_valid_mask_pixels:
                    valid_depths = valid_depths_dilated
                    valid_mask = valid_mask_dilated
                    candidate_mask_for_debug = dilated_mask_roi
                    used_mask_type = "FastSAM dilated mask"

            # Draw masks in debug image
            debug_roi = new_point_cloud[v1:v2, u1:u2]
            debug_roi[exact_mask_roi] = 80
            debug_roi[candidate_mask_for_debug] = 120
            debug_roi[valid_mask] = 255
            new_point_cloud[v1:v2, u1:u2] = debug_roi

            # ------------------------------------------------------
            # ROBUST VALIDATION
            # ------------------------------------------------------
            if valid_depths.size == 0:
                rospy.logwarn(
                    "[FastSAM DEPTH] Agujero detectado, pero SIN ningún punto 3D válido. "
                    "No se puede calcular Z, pero NO se descarta la detección 2D."
                )
                continue

            if valid_depths.size < self.min_valid_mask_pixels:
                rospy.logwarn(
                    "[FastSAM DEPTH] Pocos puntos válidos (%d/%d). "
                    "Se calcula Z igualmente para no perder el agujero.",
                    valid_depths.size,
                    self.min_valid_mask_pixels
                )

            distance_raw = float(np.median(valid_depths))
            z_min = float(np.min(valid_depths))
            z_max = float(np.max(valid_depths))
            z_p10 = float(np.percentile(valid_depths, 10))
            z_p90 = float(np.percentile(valid_depths, 90))
            z_spread = z_p90 - z_p10

            if distance_raw < self.z_min_valid or distance_raw > self.z_max_valid:
                rospy.logwarn(
                    "[FastSAM DEPTH] Z descartada por rango | "
                    "Z_raw=%.3fm | rango=[%.2f, %.2f]",
                    distance_raw,
                    self.z_min_valid,
                    self.z_max_valid
                )
                continue

            if z_spread > self.max_z_spread:
                rospy.logwarn(
                    "[FastSAM DEPTH] Z con mucha dispersión, pero NO se descarta | "
                    "Z_raw=%.3fm | p10=%.3f p90=%.3f spread=%.3fm",
                    distance_raw,
                    z_p10,
                    z_p90,
                    z_spread
                )

            distance_filtered = self._update_temporal_filter(distance_raw)

            if distance_filtered is None:
                rospy.logwarn(
                    "[FastSAM DEPTH] Filtro temporal habría descartado la Z, "
                    "pero se usa Z_raw para no perder el agujero."
                )
                distance_filtered = distance_raw

            # Calculate representative 2D point from valid mask pixels
            valid_v_local, valid_u_local = np.where(valid_mask)

            if valid_u_local.size == 0 or valid_v_local.size == 0:
                rospy.logwarn("[FastSAM DEPTH] No hay centroide válido en máscara.")
                continue

            hole_center_x_px = int(u1 + np.median(valid_u_local))
            hole_center_y_px = int(v1 + np.median(valid_v_local))

            rospy.loginfo(
                "[DEBUG DET] AGUJERO | Pixels: u(%d-%d) v(%d-%d)",
                u1,
                u2,
                v1,
                v2
            )

            rospy.loginfo(
                "[DEBUG DET] Z-Depth using %s | "
                "RawMedian: %.3fm | FilteredMedian: %.3fm | "
                "Min: %.3f Max: %.3f | P10: %.3f P90: %.3f | Count: %d",
                used_mask_type,
                distance_raw,
                distance_filtered,
                z_min,
                z_max,
                z_p10,
                z_p90,
                len(valid_depths)
            )

            # ------------------------------------------------------
            # 3D COORDINATES
            # ------------------------------------------------------
            if self.__fx == 0.0 or self.__fy == 0.0:
                rospy.logwarn(
                    "[DEBUG DET] CameraInfo todavía no recibido. No se puede calcular X,Y."
                )
                continue

            pos_x = (hole_center_x_px - self.__cx) * distance_filtered / self.__fx
            pos_y = (hole_center_y_px - self.__cy) * distance_filtered / self.__fy

            rospy.loginfo(
                "[DEBUG DET] 3D Cam Frame FILTERED: X=%.3f, Y=%.3f, Z=%.3f",
                pos_x,
                pos_y,
                distance_filtered
            )

            det_3d = Detection3D()
            det_3d.class_id = bb.class_id
            det_3d.score = bb.score
            det_3d.x = pos_x
            det_3d.y = pos_y
            det_3d.z = distance_filtered

            # Real size estimation using filtered distance
            det_3d.width = (bb.w * width * distance_filtered) / self.__fx
            det_3d.height = (bb.h * height * distance_filtered) / self.__fy

            out_msg.detections.append(det_3d)

        # Publish 3D detections
        self.pub_stereo_detect3d.publish(out_msg)

        # Publish debug image
        image_msg = self.bridge.cv2_to_imgmsg(new_point_cloud, encoding="mono8")
        image_msg.header = msg_point2.header
        self.pub_deb.publish(image_msg)


if __name__ == "__main__":
    rospy.init_node("distance_reader")
    StereoDistanceEstimator()
    rospy.spin()