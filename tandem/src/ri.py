#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Twist, Point, PointStamped, PoseStamped
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from dynamic_reconfigure.server import Server
from tandem.cfg import VerticalInspectorConfig

from cola2_msgs.msg import BodyVelocityReq, GoalDescriptor, Bool6Axis, NavSts
from net_hole_detector.msg import BoundingBoxArray


class VerticalInspector(object):
    """
    Generates a vertical inspection pattern (lawnmower in Y-Z).
    The depth reference (Z) is obtained directly from NavSts.
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
        self.vy_max = 0.1
        self.int_y_limit = 0.2
        self.tol_y = 0.15

        # Z Control
        self.kp_z = 0.12
        self.ki_z = 0.01
        self.kd_z = 0.0
        self.vz_max = 0.1
        self.int_z_limit = 0.2
        self.tol_z = 0.15

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
        # --- HOLE DETECTOR TRACKING FILTER ---
        # ==========================================
        self.hole_pose = None
        self.candidate_hole_pose = None
        self.hole_detect_count = 0
        self.last_hole_time = rospy.Time(0)

        # Parámetros del filtro espacio-temporal
        self.req_detections = 3
        self.max_hole_dist = 1.25
        self.max_hole_timeout = 9.0

        self.sub_hole = rospy.Subscriber(
            "/net_hole_detector/hole",
            PoseStamped,
            self.hole_cb,
            queue_size=1
        )

        # ==========================================
        # --- VISUAL SERVOING FINAL CON BBOX ---
        # ==========================================
        self.last_bbox = None
        self.last_bbox_time = rospy.Time(0)

        # Si hay bbox 2D consistente, podemos entrar en APPROACH_HOLE
        # aunque todavía no exista /net_hole_detector/hole 3D.
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

        # Tolerancias de centrado en imagen.
        # bbox.x y bbox.y están normalizados entre 0 y 1.
        self.visual_tol_x = rospy.get_param("~visual_tol_x", 0.08)
        self.visual_tol_y = rospy.get_param("~visual_tol_y", 0.08)

        # Ganancias visuales.
        # Antes se usaba kp_visual_yaw para girar hacia el agujero.
        # Ahora mantenemos yaw frontal y usamos vy para centrar lateralmente.
        self.kp_visual_yaw = rospy.get_param("~kp_visual_yaw", 0.8)
        self.kp_visual_y = rospy.get_param("~kp_visual_y", 0.12)
        self.kp_visual_z = rospy.get_param("~kp_visual_z", 0.25)

        # Si al corregir lateralmente se mueve al lado contrario, cambiar a -1.0 en launch.
        self.visual_y_sign = rospy.get_param("~visual_y_sign", 1.0)

        # Solo avanzar si además de estar centrado visualmente, el yaw está bien alineado.
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
        Guarda la mejor bbox actual.

        Ahora también sirve para NO perder el agujero:
        - Si hay detección 2D consistente durante varias imágenes,
          entramos en APPROACH_HOLE aunque todavía no haya Z 3D.
        - En APPROACH_HOLE el robot se centra visualmente.
        - Solo avanza cuando ya haya hole_pose 3D.
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
                "BBox visual descartada por score bajo: %.2f < %.2f",
                best_box.score,
                self.visual_min_bbox_score
            )
            return

        self.last_bbox = best_box
        self.last_bbox_time = now

        if not self.enable_visual_only_approach:
            return

        # Solo abortamos el patrón de búsqueda si estamos MOVING.
        # No lo hacemos durante INITIALIZING ni FINISHED.
        if self.state != "MOVING":
            return

        # Si pasa demasiado tiempo entre detecciones, reiniciamos racha.
        if (now - self.last_visual_detection_time).to_sec() > self.visual_detection_timeout:
            self.visual_detect_count = 0

        self.visual_detect_count += 1
        self.last_visual_detection_time = now

        rospy.loginfo_throttle(
            1.0,
            "BBox visual consistente: %d/%d | score=%.2f",
            self.visual_detect_count,
            self.req_visual_detections,
            best_box.score
        )

        if self.visual_detect_count >= self.req_visual_detections:
            rospy.logwarn(
                "!!! AGUJERO DETECTADO EN 2D !!! Entrando en APPROACH_HOLE visual aunque todavía no haya Z 3D."
            )

            self.state = "APPROACH_HOLE"
            self.visual_detect_count = 0

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

            # Force net Y-Z to be calculated based on the desired orientation
            self.yaw0 = self.target_yaw

            self.has_init = True
            rospy.loginfo("Initial position set.")
            self._generate_waypoints()

        self.last_pose = (px, py, pz, yaw)
        self.main_control_loop()

    def hole_cb(self, msg):
        """
        Recibe la posición 3D del agujero en world_ned.

        Filtro usado:
        - Si no hay candidato, crea uno.
        - Si llega una detección cercana al candidato, suma racha.
        - Si llega una detección que salta demasiado, se ignora.
        - Si pasa demasiado tiempo sin detecciones válidas, se reinicia candidato.
        - Cuando entra en APPROACH_HOLE, se congela hole_pose.
        """

        hx = msg.pose.position.x
        hy = msg.pose.position.y
        hz = msg.pose.position.z
        now = rospy.Time.now()

        new_hole_pose = (hx, hy, hz)

        # Si ya estamos terminados, ignoramos nuevas detecciones.
        if self.state in ["READY_TO_CROSS", "FINISHED"]:
            return

        # Si hemos entrado en APPROACH_HOLE solo por bbox 2D,
        # aceptamos el primer /hole 3D que llegue para poder avanzar.
        if self.state == "APPROACH_HOLE":
            if self.hole_pose is None:
                self.hole_pose = new_hole_pose
                rospy.loginfo(
                    "Z/pose 3D recibida durante APPROACH_HOLE visual. "
                    "A partir de ahora se puede usar distancia 3D para avanzar."
                )
            return

        # Solo buscamos durante ALIGNING o MOVING
        if self.state not in ["MOVING", "ALIGNING"]:
            return

        # Primera detección candidata
        if self.candidate_hole_pose is None:
            self.candidate_hole_pose = new_hole_pose
            self.hole_detect_count = 1
            self.last_hole_time = now

            rospy.loginfo(
                "Posible agujero detectado. Iniciando tracking... (1/%d)",
                self.req_detections
            )
            return

        cx, cy, cz = self.candidate_hole_pose

        time_diff = (now - self.last_hole_time).to_sec()

        dist_3d = math.sqrt(
            (hx - cx) ** 2 +
            (hy - cy) ** 2 +
            (hz - cz) ** 2
        )

        # Regla temporal
        if time_diff > self.max_hole_timeout:
            rospy.logwarn(
                "Tracking perdido por tiempo (%.1fs). Reiniciando candidato...",
                time_diff
            )

            self.candidate_hole_pose = new_hole_pose
            self.hole_detect_count = 1
            self.last_hole_time = now
            return

        # Regla espacial
        if dist_3d > self.max_hole_dist:
            rospy.logwarn(
                "Detección descartada por salto espacial (%.2fm). Se mantiene el candidato anterior.",
                dist_3d
            )
            return

        # Detección válida
        self.hole_detect_count += 1
        self.last_hole_time = now

        # Smooth candidate to reduce noise from /points2
        alpha = 0.5
        self.candidate_hole_pose = (
            (1.0 - alpha) * cx + alpha * hx,
            (1.0 - alpha) * cy + alpha * hy,
            (1.0 - alpha) * cz + alpha * hz
        )

        rospy.loginfo(
            "Agujero consistente. Racha: %d/%d",
            self.hole_detect_count,
            self.req_detections
        )

        # Agujero confirmado
        if self.hole_detect_count >= self.req_detections:
            rospy.loginfo("!!! AGUJERO 100% VERIFICADO !!! Abortando patrón de búsqueda.")

            self.hole_pose = self.candidate_hole_pose
            self.state = "APPROACH_HOLE"

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
                    "!!! SAFETY LIMIT REACHED (%.2fm) !!! Forcing ASCENT.",
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

            have_3d_target = self.hole_pose is not None

            if have_3d_target:
                hx, hy, hz = self.hole_pose

                # Distancia al objetivo 3D congelado.
                # Se usa como referencia aproximada de parada a 1.5 m.
                dist_xy = math.sqrt((hx - px) ** 2 + (hy - py) ** 2)
                dist_seguridad = 1.5
                e_dist = dist_xy - dist_seguridad
            else:
                hx = hy = hz = None
                dist_xy = None
                dist_seguridad = 1.5
                e_dist = None

            now = rospy.Time.now()

            bbox_is_recent = (
                self.last_bbox is not None and
                (now - self.last_bbox_time).to_sec() < self.bbox_timeout
            )

            # ======================================================
            # APPROACH VISUAL
            # ======================================================
            if bbox_is_recent:
                bx = self.last_bbox.x
                by = self.last_bbox.y

                # bbox.x y bbox.y están normalizados entre 0 y 1.
                # err_img_x > 0: agujero aparece a la derecha de la imagen.
                # err_img_y > 0: agujero aparece por debajo del centro.
                err_img_x = bx - 0.5
                err_img_y = by - 0.5

                centered_x = abs(err_img_x) < self.visual_tol_x
                centered_y = abs(err_img_y) < self.visual_tol_y

                # --------------------------------------------------
                # NUEVA LÓGICA:
                # - NO giramos el robot para centrar el agujero.
                # - Mantenemos yaw frontal hacia la red.
                # - Centramos horizontalmente con velocidad lateral vy.
                # --------------------------------------------------

                # Mantener yaw deseado/frontal.
                e_yaw_hold = self.normalize_angle(self.target_yaw - yaw)
                yaw_centered = abs(e_yaw_hold) < self.tol_yaw

                wz = self.clip(
                    self.kp_yaw * e_yaw_hold,
                    -self.wz_max,
                    self.wz_max
                )

                # Movimiento lateral para centrar agujero.
                vy = self.clip(
                    self.visual_y_sign * self.kp_visual_y * err_img_x,
                    -self.vy_max,
                    self.vy_max
                )

                # Si ya está centrado en X, no metas velocidad lateral residual.
                if centered_x:
                    vy = 0.0

                # Movimiento vertical para centrar en altura.
                vz = self.clip(
                    self.kp_visual_z * err_img_y,
                    -self.vz_max,
                    self.vz_max
                )

                # Si ya está centrado en Y, no metas velocidad vertical residual.
                if centered_y:
                    vz = 0.0

                vx = 0.0

                can_advance_yaw = (
                    yaw_centered or
                    (not self.require_yaw_centered_for_forward)
                )

                # Solo avanzamos si:
                # 1) agujero centrado en imagen
                # 2) yaw frontal mantenido
                # 3) tenemos objetivo 3D con distancia
                if centered_x and centered_y and can_advance_yaw:
                    if have_3d_target:
                        if e_dist > 0.15:
                            kp_x = 0.2
                            vx_max = 0.15
                            vx = self.clip(kp_x * e_dist, -vx_max, vx_max)
                        else:
                            rospy.loginfo("¡LLEGAMOS AL PUNTO DE SEGURIDAD! Listos para cruzar.")
                            self.state = "FINISHED"
                            self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_ready")
                            return
                    else:
                        rospy.loginfo_throttle(
                            1.0,
                            "Agujero centrado visualmente, pero todavía SIN Z 3D. "
                            "Se mantiene centrado y NO avanza."
                        )

                e_dist_log = e_dist if have_3d_target else float("nan")

                self.publish_cmd(vx, vy, vz, wz, "inspector_visual_lateral_approach")

                rospy.loginfo_throttle(
                    1.0,
                    "Approach VISUAL LATERAL | ImgErrX: %.2f ImgErrY: %.2f | "
                    "Centered(%s,%s) | YawErr: %.2f rad YawOK:%s | "
                    "ErrDist: %.2fm | Cmd(vx:%.2f vy:%.2f vz:%.2f wz:%.2f)",
                    err_img_x,
                    err_img_y,
                    str(centered_x),
                    str(centered_y),
                    e_yaw_hold,
                    str(yaw_centered),
                    e_dist_log,
                    vx,
                    vy,
                    vz,
                    wz
                )

                return

            # ======================================================
            # FALLBACK 3D
            # ======================================================
            # Si no hay bbox reciente y tampoco tenemos hole_pose 3D,
            # no podemos avanzar. Mantenemos yaw frontal y esperamos.
            if not have_3d_target:
                e_yaw_hold = self.normalize_angle(self.target_yaw - yaw)
                wz = self.clip(self.kp_yaw * e_yaw_hold, -self.wz_max, self.wz_max)

                self.publish_cmd(0.0, 0.0, 0.0, wz, "inspector_waiting_visual_or_3d")

                rospy.loginfo_throttle(
                    1.0,
                    "APPROACH_HOLE sin bbox reciente y sin Z 3D. "
                    "Esperando detección visual o /net_hole_detector/hole. YawErr=%.2f",
                    e_yaw_hold
                )
                return

            # Si no hay bbox reciente, usamos el método antiguo.
            # Esta era la versión estable previa a meter Z_live.

            e_z = hz - pz
            vz_cmd = self.kp_z * e_z
            vz = self.clip(vz_cmd, -self.vz_max, self.vz_max)
            reached_z = abs(e_z) < self.tol_z

            target_yaw_hole = math.atan2(hy - py, hx - px)
            e_yaw = self.normalize_angle(target_yaw_hole - yaw)
            wz_cmd = self.kp_yaw * e_yaw
            wz = self.clip(wz_cmd, -self.wz_max, self.wz_max)
            reached_yaw = abs(e_yaw) < self.tol_yaw

            vx = 0.0

            if reached_z and reached_yaw:
                if e_dist > 0.15:
                    kp_x = 0.2
                    vx_max = 0.15
                    vx = self.clip(kp_x * e_dist, -vx_max, vx_max)

                else:
                    rospy.loginfo("¡LLEGAMOS AL PUNTO DE SEGURIDAD! Listos para cruzar.")
                    self.state = "FINISHED"
                    self.publish_cmd(0.0, 0.0, 0.0, 0.0, "inspector_ready")
                    return

            self.publish_cmd(vx, 0.0, vz, wz, "inspector_approaching")

            rospy.loginfo_throttle(
                1.0,
                "Approach 3D FALLBACK | ErrDist: %.2fm | ErrZ: %.2f | ErrYaw: %.1f deg | Cmd(vx:%.2f vz:%.2f wz:%.2f)",
                e_dist,
                e_z,
                math.degrees(e_yaw),
                vx,
                vz,
                wz
            )

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