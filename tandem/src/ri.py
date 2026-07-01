#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math

from geometry_msgs.msg import Twist, Point, PointStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from dynamic_reconfigure.server import Server

from tandem.cfg import VerticalInspectorConfig

from cola2_msgs.msg import BodyVelocityReq, GoalDescriptor, Bool6Axis, NavSts
from net_hole_detector.msg import BoundingBoxArray, Detection3DArray


class VerticalInspector(object):
    """
    Generates a vertical inspection pattern (lawnmower in Y-Z).

    Final approach strategy:
    - The robot detects the hole in 2D using the left stereo bounding box.
    - It visually centers the hole using lateral velocity vy and vertical velocity vz.
    - It keeps the desired yaw fixed, looking frontally at the net.
    - Once the hole is centered, it uses the Z distance published by
      /net_hole_detector/stereo_detections_3d.
    - This Z comes from masked_stereo_z.launch.
    Some of this code was already given by the SRV Group Github.
    """

    def __init__(self):

        # ==================================================================
        # ### --- MANUAL FILTER CONFIGURATION --- ###
        self.use_smoothing_flag = False
        # ==================================================================

        # ---- MISSION Parameters (Static) ----
        self.inspection_width = rospy.get_param("~inspection_width", 10.0)
        self.inspection_depth = rospy.get_param("~inspection_depth", 15.0)
        self.step_down_z = rospy.get_param("~step_down_z", 2.0)
        self.start_z = rospy.get_param("~start_z", 0.5)

        # Orientation to look at the net (-1.57 rad = -90 degree)
        self.target_yaw = rospy.get_param("~target_yaw", -1.57)

        # Safety Limit
        self.max_safe_depth = 100

        # Yaw Control
        self.kp_yaw = 0.8
        self.wz_max = 0.3
        self.tol_yaw = 0.05

        # Y Control
        self.kp_y = 0.20
        self.ki_y = 0.0
        self.kd_y = 0.0
        self.vy_max = rospy.get_param("~vy_max", 0.1)
        self.int_y_limit = 0.2
        self.tol_y = 0.15

        # Z Control
        self.kp_z = 0.12
        self.ki_z = 0.01
        self.kd_z = 0.0
        self.vz_max = rospy.get_param("~vz_max", 0.1)
        self.int_z_limit = 0.2
        self.tol_z = 0.15

        # Forward approach control using good masked-stereo Z
        self.vx_max = rospy.get_param("~vx_max", 0.05)
        self.kp_forward = rospy.get_param("~kp_forward", 0.10)
        self.target_hole_distance = rospy.get_param("~target_hole_distance", 1.0)
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 0.08)

        # Filter parameters
        self.alpha = 0.3
        self.max_dv = 0.10

        # ---- Internal filter variables ----
        self.vy_f = 0.0
        self.vz_f = 0.0

        # ---- Topics/frames ----
        self.navigation_topic = rospy.get_param("~navigation_topic", "/girona500/navigator/navigation")
        self.frame_id = rospy.get_param("~frame_id", "girona500/base_link")
        self.rate_hz = rospy.get_param("~rate_hz", 20.0)

        # ---- Pub/Sub ----
        self.pub = rospy.Publisher(
            "/girona500/controller/body_velocity_req",
            BodyVelocityReq,
            queue_size=10
        )

        self.pub_markers = rospy.Publisher("inspection_pattern_marker", Marker, queue_size=1)

        # --- RQT PLOT PUBLISHERS ---
        self.pub_vz_out = rospy.Publisher("debug/vz_command", Float64, queue_size=1)
        self.pub_target_z = rospy.Publisher("debug/target_depth", Float64, queue_size=1)
        self.pub_real_z = rospy.Publisher("debug/real_depth", Float64, queue_size=1)
        self.pub_error_z = rospy.Publisher("debug/error_depth", Float64, queue_size=1)

        # Subscription to Navigation
        self.sub_nav = rospy.Subscriber(self.navigation_topic, NavSts, self.nav_cb, queue_size=10)
        self.pub_current_point = rospy.Publisher("current_target_point", PointStamped, queue_size=1)

        
        # ==========================================
        # --- FINAL VISUAL SERVOING USING BBOX ---
        # ==========================================
        self.last_bbox = None
        self.last_bbox_time = rospy.Time(0)

        self.enable_visual_only_approach = rospy.get_param(
            "~enable_visual_only_approach",
            True
        )
        self.req_visual_detections = rospy.get_param("~req_visual_detections", 3)
        self.visual_detection_timeout = rospy.get_param("~visual_detection_timeout", 2.0)
        self.visual_min_bbox_score = rospy.get_param("~visual_min_bbox_score", 0.20)

        self.visual_detect_count = 0
        self.last_visual_detection_time = rospy.Time(0)

        self.bbox_topic = rospy.get_param(
            "~bbox_topic",
            "/net_hole_detector/stereo_left/bounding_boxes"
        )

        self.bbox_timeout = rospy.get_param("~bbox_timeout", 1.0)

        self.visual_tol_x = rospy.get_param("~visual_tol_x", 0.08)
        self.visual_tol_y = rospy.get_param("~visual_tol_y", 0.08)

        self.kp_visual_yaw = rospy.get_param("~kp_visual_yaw", 0.8)
        self.kp_visual_y = rospy.get_param("~kp_visual_y", 0.12)
        self.kp_visual_z = rospy.get_param("~kp_visual_z", 0.25)

        self.visual_y_sign = rospy.get_param("~visual_y_sign", 1.0)

        self.require_yaw_centered_for_forward = rospy.get_param(
            "~require_yaw_centered_for_forward",
            True
        )

        self.sub_bbox = rospy.Subscriber(
            self.bbox_topic,
            BoundingBoxArray,
            self.bbox_cb,
            queue_size=1
        )

        # ==========================================
        # --- GOOD Z FROM masked_stereo_z.launch ---
        # ==========================================
        self.stereo_3d_topic = rospy.get_param(
            "~stereo_3d_topic",
            "/net_hole_detector/stereo_detections_3d"
        )

        self.stereo_3d_timeout = rospy.get_param("~stereo_3d_timeout", 1.0)

        self.current_hole_z = float("nan")
        self.last_hole_z_time = rospy.Time(0)

        self.sub_stereo_3d = rospy.Subscriber(
            self.stereo_3d_topic,
            Detection3DArray,
            self.stereo_3d_cb,
            queue_size=1
        )

        # ---- States ----
        self.state = "INITIALIZING"
        self.waypoints = []
        self.current_wp_index = 0

        self.has_init = False
        self.x0 = self.y0 = self.z0 = 0.0
        self.yaw0 = 0.0
        self.last_pose = None
        self.start_time = rospy.Time.now()

        # Internal Integral accumulators
        self.int_z = 0.0
        self.int_y = 0.0
        self.vx_est = 0.0
        self.vy_est = 0.0
        self.vz_est = 0.0

        # ---- DYNAMIC RECONFIGURE SERVER ----
        self.srv = Server(VerticalInspectorConfig, self.reconfigure_cb)

        rospy.loginfo("VerticalInspector ready | nav_topic=%s", self.navigation_topic)
        rospy.loginfo("FILTERING ENABLED: %s", self.use_smoothing_flag)
        rospy.loginfo("SAFETY LIMIT Z: %.2f m", self.max_safe_depth)
        rospy.loginfo("Visual bbox topic: %s", self.bbox_topic)
        rospy.loginfo("Stereo 3D topic for final approach: %s", self.stereo_3d_topic)
        rospy.loginfo(
            "Final approach target distance: %.2f m | tolerance: %.2f m",
            self.target_hole_distance,
            self.distance_tolerance
        )

    def reconfigure_cb(self, config, level):
        """Dynamic reconfigure callback."""

        # Y Control
        self.kp_y = config.kp_y
        if hasattr(config, 'ki_y'):
            self.ki_y = config.ki_y
        if hasattr(config, 'kd_y'):
            self.kd_y = config.kd_y
        if hasattr(config, 'int_y_limit'):
            self.int_y_limit = config.int_y_limit

        self.vy_max = config.vy_max
        self.tol_y = config.tol_y

        # Z Control
        self.kp_z = config.kp_z
        self.ki_z = config.ki_z
        if hasattr(config, 'kd_z'):
            self.kd_z = config.kd_z

        self.vz_max = config.vz_max
        self.int_z_limit = config.int_z_limit
        self.tol_z = config.tol_z

        # Filter values
        self.alpha = config.alpha_filter
        self.max_dv = config.max_dv

        return config

    def _generate_waypoints(self):
        """Generates the list of relative waypoints (dy, dz) and publishes them to RViz."""

        self.waypoints = []

        half_width = self.inspection_width / 2.0
        current_z = self.start_z
        direction = 1
        start_y = -half_width

        self.waypoints.append((start_y, current_z))

        while current_z < self.inspection_depth:
            target_y = direction * half_width
            self.waypoints.append((target_y, current_z))

            current_z += self.step_down_z

            if current_z > self.inspection_depth:
                current_z = self.inspection_depth

            self.waypoints.append((target_y, current_z))

            direction *= -1

            if current_z == self.inspection_depth:
                target_y = direction * half_width
                self.waypoints.append((target_y, current_z))
                break

        rospy.loginfo("Generated %d waypoints.", len(self.waypoints))

        if self.has_init:
            self.publish_path_line_marker()

    def publish_path_line_marker(self):
        """Publishes the generated waypoints as a LINE_STRIP Marker."""

        marker = Marker()
        marker.header.frame_id = "world_ned"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "inspection_pattern"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        cy = math.cos(self.yaw0)
        sy = math.sin(self.yaw0)

        for (dy, dz) in self.waypoints:
            p = Point()
            p.x = self.x0 - sy * dy
            p.y = self.y0 + cy * dy
            p.z = self.z0 + dz
            marker.points.append(p)

        self.pub_markers.publish(marker)
        rospy.loginfo("Published inspection pattern marker to RViz.")

    def publish_current_target_point(self, current_wp_index):
        """Publishes the current waypoint target as PointStamped."""

        dy, dz = self.waypoints[current_wp_index]

        point_stamped = PointStamped()
        point_stamped.header.frame_id = "world_ned"
        point_stamped.header.stamp = rospy.Time.now()

        cy = math.cos(self.yaw0)
        sy = math.sin(self.yaw0)

        point_stamped.point.x = self.x0 - sy * dy
        point_stamped.point.y = self.y0 + cy * dy
        point_stamped.point.z = self.z0 + dz

        self.pub_current_point.publish(point_stamped)

    def bbox_cb(self, msg):
        """
        Stores the current best 2D bounding box.

        If there is a consistent visual detection, the robot enters
        APPROACH_HOLE even if the final stereo Z has not arrived yet.
        """

        now = rospy.Time.now()

        if not msg.boxes:
            if (now - self.last_bbox_time).to_sec() > self.bbox_timeout:
                self.visual_detect_count = 0
            return

        best_box = max(msg.boxes, key=lambda b: b.score)

        if best_box.score < self.visual_min_bbox_score:
            rospy.logwarn_throttle(
                1.0,
                "BBox Visual discarded due to low score: %.2f < %.2f",
                best_box.score,
                self.visual_min_bbox_score
            )
            return

        self.last_bbox = best_box
        self.last_bbox_time = now

        if not self.enable_visual_only_approach:
            return

        if self.state != "MOVING":
            return

        if (now - self.last_visual_detection_time).to_sec() > self.visual_detection_timeout:
            self.visual_detect_count = 0

        self.visual_detect_count += 1
        self.last_visual_detection_time = now

        rospy.loginfo_throttle(
            1.0,
            "Consistent visual BBox: %d/%d | score=%.2f",
            self.visual_detect_count,
            self.req_visual_detections,
            best_box.score
        )

        if self.visual_detect_count >= self.req_visual_detections:
            rospy.logwarn(
                "¡¡¡ HOLE DETECTED IN 2D !!! Entering visual APPROACH_HOLE."
            )

            self.state = "APPROACH_HOLE"
            self.visual_detect_count = 0

    def stereo_3d_cb(self, msg):
        """
        Receives the final Z calculated by masked_stereo_z.launch.

        Topic:
            /net_hole_detector/stereo_detections_3d

        Used value:
            msg.detections[0].z

        This is the distance from the stereo camera to the hole.
        It is not transformed to world_ned.
        """

        if not msg.detections:
            return

        det = msg.detections[0]

        if det.z <= 0.0:
            return

        self.current_hole_z = float(det.z)
        self.last_hole_z_time = rospy.Time.now()

    def nav_cb(self, msg):
        """
        Navigation callback.
        Executes the control loop.
        """

        px = msg.position.north
        py = msg.position.east
        pz = msg.position.depth
        yaw = msg.orientation.yaw

        self.vx_est = msg.body_velocity.x
        self.vy_est = msg.body_velocity.y
        self.vz_est = msg.body_velocity.z

        if not self.has_init:
            self.x0, self.y0, self.z0, self.yaw0 = px, py, pz, yaw

            self.yaw0 = self.target_yaw

            self.has_init = True
            rospy.loginfo("Initial position set.")
            self._generate_waypoints()

        self.last_pose = (px, py, pz, yaw)
        self.main_control_loop()

    
    def main_control_loop(self):
        """Main controller state machine."""

        # --- State 1: INITIALIZING ---
        if self.state == "INITIALIZING":
            is_ready = self.has_init and self.last_pose is not None

            if not is_ready:
                if not self.has_init:
                    rospy.logwarn_throttle(2.0, "Waiting for odometry...")

                self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_initializing")
                return

            else:
                rospy.loginfo("Initialization complete. Aligning robot.")
                self.state = "ALIGNING"
                self.current_wp_index = 0

        # --- State 2: ALIGNING ---
        elif self.state == "ALIGNING":
            px, py, pz, yaw = self.last_pose

            e_yaw = self.normalize_angle(self.target_yaw - yaw)
            wz_cmd = self.kp_yaw * e_yaw
            wz = self.clip(wz_cmd, -self.wz_max, self.wz_max)

            if abs(e_yaw) < self.tol_yaw:
                rospy.loginfo("Alignment complete! Starting inspection pattern.")
                self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_aligned")
                self.state = "MOVING"
                self.current_wp_index = 0

            else:
                self.publish_cmd(0.0, 0.0, 0.0, wz, "inspector_aligning")
                rospy.loginfo_throttle(
                    1.0,
                    "Aligning | ErrYaw: %.2f rad | Cmd(wz: %.2f)",
                    e_yaw,
                    wz
                )

        # --- State 3: MOVING ---
        elif self.state == "MOVING":

            if self.current_wp_index >= len(self.waypoints):
                rospy.loginfo("Mission completed.")
                self.state = "FINISHED"
                self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_finished")
                return

            target_dy, target_dz = self.waypoints[self.current_wp_index]
            px, py, pz, yaw = self.last_pose

            target_depth = self.z0 + target_dz
            current_depth = pz

            # ---------------- ERROR IN Y AXIS ----------------
            dx_w = px - self.x0
            dy_w = py - self.y0

            cy = math.cos(self.yaw0)
            sy = math.sin(self.yaw0)

            dy_b = -sy * dx_w + cy * dy_w
            e_y = target_dy - dy_b

            vy_cmd = (self.kp_y * e_y - self.kd_y * self.vy_est + self.int_y)

            if abs(vy_cmd) < self.vy_max - 1e-3:
                self.int_y += self.ki_y * e_y / self.rate_hz
                self.int_y = self.clip(self.int_y, -self.int_y_limit, self.int_y_limit)

            vy = self.clip(vy_cmd, -self.vy_max, self.vy_max)
            reached_y = abs(e_y) < self.tol_y

            # ---------------- ERROR IN Z AXIS ----------------
            if current_depth > self.max_safe_depth:
                rospy.logwarn_throttle(
                    1.0,
                    "¡¡¡ SAFETY LIMIT REACHED (%.2fm) !!! Forcing ASCENT.",
                    current_depth
                )

                vz = -0.9
                reached_z = False
                self.int_z = 0.0
                e_z_or_depth = 0.0

            else:
                e_depth = target_depth - current_depth
                e_z_or_depth = e_depth

                vz_cmd = (self.kp_z * e_z_or_depth - self.kd_z * self.vz_est + self.int_z)

                if abs(vz_cmd) < self.vz_max - 1e-3:
                    self.int_z += self.ki_z * e_z_or_depth / self.rate_hz
                    self.int_z = self.clip(self.int_z, -self.int_z_limit, self.int_z_limit)

                vz = self.clip(vz_cmd, -self.vz_max, self.vz_max)
                reached_z = abs(e_z_or_depth) < self.tol_z

            # Waypoint transition
            if reached_y and reached_z:
                rospy.loginfo("WP %d reached.", self.current_wp_index)
                self.current_wp_index += 1

                self.int_z = 0.0
                self.int_y = 0.0

                vy = 0.0
                vz = 0.0

            # Apply smoothing if enabled
            vy_out, vz_out = self.smooth(vy, vz)

            # Heading hold while moving
            e_yaw_moving = self.normalize_angle(self.target_yaw - yaw)
            wz_cmd = self.kp_yaw * e_yaw_moving
            wz_out = self.clip(wz_cmd, -self.wz_max, self.wz_max)

            requester = "inspector_wp_{}".format(self.current_wp_index)
            self.publish_cmd(0.0, vy_out, vz_out, wz_out, requester)

            # RQT plot
            self.pub_real_z.publish(Float64(current_depth))
            self.pub_target_z.publish(Float64(target_depth))
            self.pub_vz_out.publish(Float64(vz_out * 10))
            self.pub_error_z.publish(Float64(e_z_or_depth))

            if self.current_wp_index < len(self.waypoints):
                self.publish_current_target_point(self.current_wp_index)

            rospy.loginfo_throttle(
                1.0,
                "WP %d | ErrY:%.2f (IntY:%.2f) | ErrZ:%.2f | Cmd(vy:%.2f, vz:%.2f)",
                self.current_wp_index,
                e_y,
                self.int_y,
                e_z_or_depth,
                vy_out,
                vz_out
            )

        # --- State 4: APPROACH_HOLE ---
        elif self.state == "APPROACH_HOLE":

            px, py, pz, yaw = self.last_pose
            now = rospy.Time.now()

            bbox_is_recent = (
                self.last_bbox is not None and
                (now - self.last_bbox_time).to_sec() < self.bbox_timeout
            )

            has_valid_stereo_z = (
                math.isfinite(self.current_hole_z) and
                (now - self.last_hole_z_time).to_sec() < self.stereo_3d_timeout
            )

            if bbox_is_recent:
                bx = self.last_bbox.x
                by = self.last_bbox.y

                # bbox.x and bbox.y are normalized in [0, 1].
                err_img_x = bx - 0.5
                err_img_y = by - 0.5

                centered_x = abs(err_img_x) < self.visual_tol_x
                centered_y = abs(err_img_y) < self.visual_tol_y

                # Keep frontal yaw to the net.
                e_yaw_hold = self.normalize_angle(self.target_yaw - yaw)
                yaw_centered = abs(e_yaw_hold) < self.tol_yaw

                wz = self.clip(
                    self.kp_yaw * e_yaw_hold,
                    -self.wz_max,
                    self.wz_max
                )

                # Lateral visual centering.
                vy = self.clip(
                    self.visual_y_sign * self.kp_visual_y * err_img_x,
                    -self.vy_max,
                    self.vy_max
                )

                if centered_x:
                    vy = 0.0

                # Vertical visual centering.
                vz = self.clip(
                    self.kp_visual_z * err_img_y,
                    -self.vz_max,
                    self.vz_max
                )

                if centered_y:
                    vz = 0.0

                vx = 0.0
                err_dist = float("nan")

                can_advance_yaw = (
                    yaw_centered or
                    (not self.require_yaw_centered_for_forward)
                )

                if centered_x and centered_y and can_advance_yaw:
                    if has_valid_stereo_z:
                        err_dist = self.current_hole_z - self.target_hole_distance

                        if abs(err_dist) <= self.distance_tolerance:
                            rospy.loginfo(
                                "¡¡¡SAFETY DISTANCE REACHED!!! "
                                "Z=%.2fm | target=%.2fm",
                                self.current_hole_z,
                                self.target_hole_distance
                            )

                            self.state = "FINISHED"
                            self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_ready")
                            return

                        else:
                            # Only move forward or stop. Do not move backwards.
                            vx = self.kp_forward * err_dist
                            vx = self.clip(vx, 0.0, self.vx_max)

                    else:
                        rospy.loginfo_throttle(
                            1.0,
                            "Visually centered hole, but still WITHOUT 3D Z from masked_stereo_z."
                            "It remains centered and DOES NOT advance."
                        
                        )

                self.publish_cmd(vx, vy, vz, wz, "inspector_visual_lateral_approach")

                rospy.loginfo_throttle(
                    1.0,
                    "Approach VISUAL LATERAL + MASKED Z | ImgErrX: %.2f ImgErrY: %.2f | "
                    "Centered(%s,%s) | YawErr: %.2f rad YawOK:%s | "
                    "Z: %.2fm Target: %.2fm ErrDist: %.2fm | "
                    "Cmd(vx:%.2f vy:%.2f vz:%.2f wz:%.2f)",
                    err_img_x,
                    err_img_y,
                    str(centered_x),
                    str(centered_y),
                    e_yaw_hold,
                    str(yaw_centered),
                    self.current_hole_z if has_valid_stereo_z else float("nan"),
                    self.target_hole_distance,
                    err_dist,
                    vx,
                    vy,
                    vz,
                    wz
                )

                return

            # If no recent bbox is available, keep yaw and wait.
            e_yaw_hold = self.normalize_angle(self.target_yaw - yaw)
            wz = self.clip(self.kp_yaw * e_yaw_hold, -self.wz_max, self.wz_max)

            self.publish_cmd(0.0, 0.0, 0.0, wz, "inspector_waiting_visual")

            rospy.loginfo_throttle(
                1.0,
                "APPROACH_HOLE without recent bbox. Waiting for visual detection. YawErr=%.2f",
                e_yaw_hold
            )

            return

        # --- State 5: FINISHED ---
        elif self.state == "FINISHED":
            vy, vz = self.smooth(0.0, 0.0)
            self.publish_cmd(0.0, vy, vz, 0.0, "inspector_finished")

        else:
            vy, vz = self.smooth(0.0, 0.0)
            self.publish_cmd(0.0, vy, vz, 0.0, "inspector_idle")

    def smooth(self, vy, vz):
        if not self.use_smoothing_flag:
            self.vy_f = vy
            self.vz_f = vz
            return vy, vz

        vy_f = self.alpha * vy + (1.0 - self.alpha) * self.vy_f
        vz_f = self.alpha * vz + (1.0 - self.alpha) * self.vz_f

        vy_f = self.clip(vy_f, self.vy_f - self.max_dv, self.vy_f + self.max_dv)
        vz_f = self.clip(vz_f, self.vz_f - self.max_dv, self.vz_f + self.max_dv)

        self.vy_f = vy_f
        self.vz_f = vz_f

        return vy_f, vz_f

    def publish_cmd(self, vx, vy, vz, wz, requester):
        """Publishes body velocity command."""

        cmd = BodyVelocityReq()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.frame_id

        gd = GoalDescriptor()
        gd.requester = requester
        gd.priority = 0
        cmd.goal = gd

        tw = Twist()
        tw.linear.x = vx
        tw.linear.y = vy
        tw.linear.z = vz
        tw.angular.z = wz
        cmd.twist = tw

        ba = Bool6Axis()
        ba.x = False
        ba.y = False
        ba.z = False
        ba.roll = True
        ba.pitch = True
        ba.yaw = False
        cmd.disable_axis = ba

        self.pub.publish(cmd)

    @staticmethod
    def normalize_angle(angle):
        """Keeps angle between -pi and pi."""

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    @staticmethod
    def clip(v, vmin, vmax):
        return max(vmin, min(v, vmax))

    @staticmethod
    def sign(v):
        return 1.0 if v >= 0.0 else -1.0


if __name__ == "__main__":
    rospy.init_node("vertical_inspector")

    try:
        VerticalInspector()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
        