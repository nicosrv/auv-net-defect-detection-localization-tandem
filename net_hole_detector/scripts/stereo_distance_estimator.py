#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from net_hole_detector.msg import BoundingBoxArray, Detection3D, Detection3DArray


class StereoDistanceEstimator:

    def __init__(self):
        self.bridge = CvBridge()

        
        # Output of stereo_image_proc applied to masked images.
        self.disparity_topic = rospy.get_param("~disparity_topic", "/masked_stereo/disparity")

        self.left_mask_topic = rospy.get_param("~left_mask_topic", "/net_hole_detector/stereo_left/hole_mask")
        self.right_mask_topic = rospy.get_param("~right_mask_topic", "/net_hole_detector/stereo_right/hole_mask")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/net_hole_detector/stereo_left/bounding_boxes")

        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic",
            "/girona500/xiroi/stereo_ch3/left_optical/camera_info"
        )

        self.baseline_fallback = float(rospy.get_param("~baseline_fallback", 0.10))

        self.min_bbox_score = float(rospy.get_param("~min_bbox_score", 0.20))
        self.min_points = int(rospy.get_param("~min_points", 20))

        # Mask 63/3.
        self.dilation_kernel_size = int(rospy.get_param("~dilation_kernel_size", 63))
        self.dilation_iterations = int(rospy.get_param("~dilation_iterations", 3))
        self.inner_exclusion_kernel = int(rospy.get_param("~inner_exclusion_kernel", 15))

        self.bbox_expand = float(rospy.get_param("~bbox_expand", 1.8))

        # Reasonable physical range.
        self.z_min_valid = float(rospy.get_param("~z_min_valid", 0.15))
        self.z_max_valid = float(rospy.get_param("~z_max_valid", 8.0))

        self.min_valid_disparity = float(rospy.get_param("~min_valid_disparity", 0.5))
        self.max_valid_disparity = float(rospy.get_param("~max_valid_disparity", 250.0))

       # Selection close to the mask prior.
        self.prior_tolerance_px = float(rospy.get_param("~prior_tolerance_px", 3.0))
        self.min_prior_points = int(rospy.get_param("~min_prior_points", 5))

        # IMPORTANT: defined to avoid crash.
        self.disp_bin_width = float(rospy.get_param("~disp_bin_width", 1.0))
        self.min_peak_points = int(rospy.get_param("~min_peak_points", 3))

        
        # If stereo_image_proc does not give points close to the prior, we use the dynamic prior.
        # It is not a fixed scale: it changes with each frame.

        self.use_prior_fallback = rospy.get_param("~use_prior_fallback", True)

        # Temporal filter to avoid sudden jumps.
        self.max_temporal_jump = float(rospy.get_param("~max_temporal_jump", 0.45))
        self.history_timeout = float(rospy.get_param("~history_timeout", 3.0))

        self.dilation_kernel_size = self.odd(self.dilation_kernel_size, 3)
        self.inner_exclusion_kernel = self.odd(self.inner_exclusion_kernel, 3)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_z = None
        self.last_disp = None
        self.last_valid_time = rospy.Time(0)

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.info_cb, queue_size=1)

        sub_disp = Subscriber(self.disparity_topic, DisparityImage)
        sub_left_mask = Subscriber(self.left_mask_topic, Image)
        sub_right_mask = Subscriber(self.right_mask_topic, Image)
        sub_bbox = Subscriber(self.bbox_topic, BoundingBoxArray)

        self.sync = ApproximateTimeSynchronizer(
            [sub_disp, sub_left_mask, sub_right_mask, sub_bbox],
            queue_size=10,
            slop=0.3
        )
        self.sync.registerCallback(self.cb)

        self.pub_det = rospy.Publisher(
            "/net_hole_detector/stereo_detections_3d",
            Detection3DArray,
            queue_size=1
        )

        self.pub_debug = rospy.Publisher(
            "/stereo/deb_image",
            Image,
            queue_size=1
        )

        rospy.loginfo("[StereoDistanceEstimator] Ready.")
        rospy.loginfo("[StereoDistanceEstimator] Source: /masked_stereo/disparity.")
        rospy.loginfo("[StereoDistanceEstimator] Method: Disparity close to the mask prior; dynamic fallback if there are no pixels.")

    @staticmethod
    def odd(v, minimum):
        v = max(int(v), minimum)
        if v % 2 == 0:
            v += 1
        return v

    def info_cb(self, msg):
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

    def resize_mask(self, mask, shape):
        h, w = shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask > 0

    def dilate(self, mask_bool, kernel_size, iterations):
        k = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        u8 = (mask_bool.astype(np.uint8)) * 255
        return cv2.dilate(u8, k, iterations=iterations) > 0

    def valid_area(self, mask, shape):
        base = self.resize_mask(mask, shape)

        if np.count_nonzero(base) == 0:
            return base

        outer = self.dilate(base, self.dilation_kernel_size, self.dilation_iterations)
        inner = self.dilate(base, self.inner_exclusion_kernel, 1)

       
        # Corona around the hole. The interior/background is avoided.
        return outer & (~inner)

    def bbox_mask(self, bb, width, height):
        cx = bb.x * width
        cy = bb.y * height
        bw = bb.w * width * self.bbox_expand
        bh = bb.h * height * self.bbox_expand

        u1 = int(cx - bw / 2.0)
        u2 = int(cx + bw / 2.0)
        v1 = int(cy - bh / 2.0)
        v2 = int(cy + bh / 2.0)

        u1 = max(0, min(width - 1, u1))
        u2 = max(0, min(width, u2))
        v1 = max(0, min(height - 1, v1))
        v2 = max(0, min(height, v2))

        m = np.zeros((height, width), dtype=bool)

        if u2 > u1 and v2 > v1:
            m[v1:v2, u1:u2] = True

        return m

    def mask_center_u(self, mask, shape):
        base = self.resize_mask(mask, shape)

        ys, xs = np.where(base)

        if xs.size < 10:
            return None

        return float(np.median(xs))

    def choose_disparity_near_prior(self, disp_values, prior_disp):
        d = np.asarray(disp_values, dtype=np.float32)
        d = d[np.isfinite(d)]
        d = d[(d > self.min_valid_disparity) & (d < self.max_valid_disparity)]

        if d.size < self.min_points:
            return None, 0, "no_points"

        if prior_disp is None or prior_disp <= self.min_valid_disparity:
            return None, 0, "no_prior"

        low = float(prior_disp - self.prior_tolerance_px)
        high = float(prior_disp + self.prior_tolerance_px)

        d_close = d[(d >= low) & (d <= high)]

        # GOOD CASE: stereo_image_proc does have disparities close to the prior.
        if d_close.size >= self.min_prior_points:
            bins = np.arange(low, high + self.disp_bin_width, self.disp_bin_width)

            if bins.size >= 2:
                hist, edges = np.histogram(d_close, bins=bins)

                candidates = []

                for i, count in enumerate(hist):
                    if count >= self.min_peak_points:
                        b_low = float(edges[i])
                        b_high = float(edges[i + 1])
                        vals = d_close[(d_close >= b_low) & (d_close < b_high)]

                        if vals.size > 0:
                            candidates.append({
                                "disp": float(np.median(vals)),
                                "count": int(vals.size),
                                "low": b_low,
                                "high": b_high
                            })

                if candidates:
                    # We choose the bin closest to the prior, not the furthest nor the most populated.

                    best = min(candidates, key=lambda c: abs(c["disp"] - prior_disp))
                    return best["disp"], int(best["count"]), "stereo_near_prior_bin"

            # If there is no clear bin, we use the median within the strict window.
            return float(np.median(d_close)), int(d_close.size), "stereo_near_prior_median"

        # FALLBACK CASE:
        # If /masked_stereo/disparity does not have any pixel close to the prior,
        # we do not choose 55, 2, 15, or other absurdities.
        # We use the dynamic prior from the left-right masks.

        if self.use_prior_fallback:
            return float(prior_disp), int(d_close.size), "mask_prior_fallback"

        return None, int(d_close.size), "not_enough_near_prior"

    def temporal_accept(self, z, disp):
        now = rospy.Time.now()
        z = float(z)
        disp = float(disp)

        if z < self.z_min_valid or z > self.z_max_valid:
            rospy.logwarn(
                "[StereoDistanceEstimator] Z out of physical range | Z=%.3f m | disp=%.2f",
                z,
                disp
            )
            return None

        if self.last_z is None or (now - self.last_valid_time).to_sec() > self.history_timeout:
            self.last_z = z
            self.last_disp = disp
            self.last_valid_time = now
            return z

        jump = abs(z - self.last_z)

        if jump > self.max_temporal_jump:
            rospy.logwarn(
                "[StereoDistanceEstimator] Z rejected due to jump | raw=%.3f prev=%.3f jump=%.3f disp=%.2f",
                z,
                self.last_z,
                jump,
                disp
            )
            return None

        self.last_z = z
        self.last_disp = disp
        self.last_valid_time = now

        return z

    def cb(self, disp_msg, left_mask_msg, right_mask_msg, bbox_msg):
        out_msg = Detection3DArray()
        out_msg.header = disp_msg.header
        out_msg.detections = []

        if self.fx is None or self.fy is None:
            self.pub_det.publish(out_msg)
            return

        if not bbox_msg.boxes:
            self.pub_det.publish(out_msg)
            return

        try:
            disp = self.bridge.imgmsg_to_cv2(disp_msg.image, desired_encoding="passthrough")
            disp = np.asarray(disp, dtype=np.float32)
        except Exception as e:
            rospy.logerr("[StereoDistanceEstimator] Error reading disparity image: %s", str(e))
            self.pub_det.publish(out_msg)
            return

        height, width = disp.shape[:2]
        shape = (height, width)

        try:
            left_mask = self.bridge.imgmsg_to_cv2(left_mask_msg, desired_encoding="mono8")
            right_mask = self.bridge.imgmsg_to_cv2(right_mask_msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr("[StereoDistanceEstimator] Error reading masks: %s", str(e))
            self.pub_det.publish(out_msg)
            return

        best_box = max(bbox_msg.boxes, key=lambda b: b.score)

        if best_box.score < self.min_bbox_score:
            self.pub_det.publish(out_msg)
            return

        left_u = self.mask_center_u(left_mask, shape)
        right_u = self.mask_center_u(right_mask, shape)

        prior_disp = None

        if left_u is not None and right_u is not None:
            prior_disp = abs(left_u - right_u)

        left_area = self.valid_area(left_mask, shape)
        box = self.bbox_mask(best_box, width, height)

        valid_mask = (
            left_area &
            box &
            np.isfinite(disp) &
            (disp > self.min_valid_disparity) &
            (disp < self.max_valid_disparity)
        )

        ys, xs = np.where(valid_mask)

        if xs.size < self.min_points:
            rospy.logwarn_throttle(
                1.0,
                "[StereoDistanceEstimator] Few valid pixels: %d/%d",
                xs.size,
                self.min_points
            )
            self.pub_det.publish(out_msg)
            return

        d_values = disp[ys, xs]

        d_sel, close_count, mode = self.choose_disparity_near_prior(d_values, prior_disp)

        rospy.loginfo_throttle(
            0.5,
            "[StereoDistanceEstimator] prior=%.2f px | close_count=%d | d_sel=%.2f | mode=%s",
            float(prior_disp) if prior_disp is not None else -1.0,
            int(close_count),
            float(d_sel) if d_sel is not None else -1.0,
            mode
        )

        if d_sel is None:
            self.pub_det.publish(out_msg)
            return

        f = float(disp_msg.f) if abs(float(disp_msg.f)) > 1e-9 else self.fx
        T = abs(float(disp_msg.T)) if abs(float(disp_msg.T)) > 1e-9 else self.baseline_fallback

        z_raw = (f * T) / d_sel
        z_final = self.temporal_accept(z_raw, d_sel)

        if z_final is None:
            self.pub_det.publish(out_msg)
            return

        selected = np.abs(d_values - d_sel) <= max(self.prior_tolerance_px, 2.0)

        if np.count_nonzero(selected) < self.min_points:
            selected = np.ones_like(d_values, dtype=bool)

        sel_x = xs[selected]
        sel_y = ys[selected]

        cx_px = float(np.median(sel_x))
        cy_px = float(np.median(sel_y))

        x_cam = (cx_px - self.cx) * z_final / self.fx
        y_cam = (cy_px - self.cy) * z_final / self.fy

        det = Detection3D()
        det.class_id = best_box.class_id
        det.score = best_box.score
        det.x = float(x_cam)
        det.y = float(y_cam)
        det.z = float(z_final)
        det.width = float((best_box.w * width * z_final) / self.fx)
        det.height = float((best_box.h * height * z_final) / self.fy)

        out_msg.detections.append(det)
        self.pub_det.publish(out_msg)

        debug = np.zeros((height, width), dtype=np.uint8)
        debug[left_area & box] = 80
        debug[sel_y, sel_x] = 255

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="mono8")
        debug_msg.header = disp_msg.header
        self.pub_debug.publish(debug_msg)

        rospy.loginfo_throttle(
            0.5,
            "[StereoDistanceEstimator] disp=%.2f px | f=%.2f T=%.3f | Z_raw=%.3f Z=%.3f | mode=%s",
            d_sel,
            f,
            T,
            z_raw,
            z_final,
            mode
        )


if __name__ == "__main__":
    rospy.init_node("stereo_distance_estimator")
    StereoDistanceEstimator()
    rospy.spin()
