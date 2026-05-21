#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from geometry_msgs.msg import Twist, Point, PointStamped
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from dynamic_reconfigure.server import Server
from tandem.cfg import VerticalInspectorConfig

from cola2_msgs.msg import BodyVelocityReq, GoalDescriptor, Bool6Axis, NavSts
# NOTA: Ya no se necesita Range

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

        # Safety Limit
        self.max_safe_depth = 100 

        # Y Control (PID)
        self.kp_y = 0.20
        self.ki_y = 0.0  
        self.vy_max = 0.1
        self.int_y_limit = 0.2
        self.tol_y = 0.15

        # Z Control (PID)
        self.kp_z = 0.12
        self.ki_z = 0.01
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
        self.pub = rospy.Publisher("/girona500/controller/body_velocity_req",
                                   BodyVelocityReq, queue_size=10)
        self.pub_markers = rospy.Publisher("inspection_pattern_marker", Marker, queue_size=1)
        
        # --- RQT PLOT PUBLISHERS ---
        self.pub_vz_out = rospy.Publisher("debug/vz_command", Float64, queue_size=1)
        self.pub_target_z = rospy.Publisher("debug/target_depth", Float64, queue_size=1)
        self.pub_real_z = rospy.Publisher("debug/real_depth", Float64, queue_size=1)
        self.pub_error_z = rospy.Publisher("debug/error_depth", Float64, queue_size=1)
        
        # Subscription to Navigation
        self.sub_nav = rospy.Subscriber(self.navigation_topic, NavSts, self.nav_cb, queue_size=10)
        self.pub_current_point = rospy.Publisher("current_target_point", PointStamped, queue_size=1)


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

    def reconfigure_cb(self, config, level):
        """ Dynamic reconfigure callback. """
        # Y Control
        self.kp_y = config.kp_y
        if hasattr(config, 'ki_y'): 
            self.ki_y = config.ki_y
        if hasattr(config, 'int_y_limit'):
            self.int_y_limit = config.int_y_limit
        if hasattr(config, 'kd_y'): 
            self.kd_y = config.kd_y
            
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
        """ Generates the list of relative waypoints (dy, dz) and publishes them to RViz. """
        self.waypoints = []
        half_width = self.inspection_width / 2.0
        current_z = 2.0
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
        """ Publishes the generated waypoints as a LINE_STRIP Marker. """
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
        """ Publica el waypoint objetivo actual como un mensaje PointStamped. """
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

    def nav_cb(self, msg):
        """
        Callback de Navegación (NavSts). Ejecuta el bucle de control.
        """
        # Extraer Posición (NED)
        px = msg.position.north
        py = msg.position.east
        pz = msg.position.depth 
        yaw = msg.orientation.yaw 

        self.vx_est = msg.body_velocity.x
        self.vy_est = msg.body_velocity.y
        self.vz_est = msg.body_velocity.z


        if not self.has_init:
            self.x0, self.y0, self.z0, self.yaw0 = px, py, pz, yaw
            self.has_init = True
            rospy.loginfo("Initial position set.")
            self._generate_waypoints()

        self.last_pose = (px, py, pz, yaw)
        self.main_control_loop()


    def main_control_loop(self):
        """ Contiene la lógica principal del controlador y la máquina de estados. """
        
        # --- State 1: INITIALIZING ---
        if self.state == "INITIALIZING":
            is_ready = self.has_init and self.last_pose is not None
                
            if not is_ready:
                if not self.has_init:
                    rospy.logwarn_throttle(2.0, "Waiting for odometry...")
                self.publish_cmd(0.0, 0.0, 0.0, "inspector_initializing")
                return
            else:
                rospy.loginfo("Initialization complete. Starting.")
                self.state = "MOVING"
                self.current_wp_index = 0

        # --- State 2: MOVING ---
        elif self.state == "MOVING":
            if self.current_wp_index >= len(self.waypoints):
                rospy.loginfo("Mission completed.")
                self.state = "FINISHED"
                self.publish_cmd(0.0, 0.0, 0.0, "inspector_finished")
                return

            target_dy, target_dz = self.waypoints[self.current_wp_index]
            px, py, pz, yaw = self.last_pose
            
            # --- CALCULATE TARGET DEPTH ---
            # La profundidad objetivo es la Profundidad Inicial (z0) + Desplazamiento (target_dz)
            target_depth = self.z0 + target_dz
            current_depth = pz # Z real es la Z de la navegación

            # ---------------- ERROR IN Y AXIS ----------------
            dx_w = px - self.x0
            dy_w = py - self.y0
            cy = math.cos(self.yaw0); sy = math.sin(self.yaw0)
            dy_b = -sy*dx_w + cy*dy_w 
            e_y = target_dy - dy_b
            
            # PI D Control for Y
            vy_cmd = (self.kp_y * e_y - self.kd_y * self.vy_est + self.int_y)

            # Anti-windup for Y
            if abs(vy_cmd) < self.vy_max - 1e-3:
                 self.int_y += self.ki_y * e_y / self.rate_hz
                 self.int_y = self.clip(self.int_y, -self.int_y_limit, self.int_y_limit)

            vy = self.clip(vy_cmd, -self.vy_max, self.vy_max)
            reached_y = abs(e_y) < self.tol_y
            
            # ---------------- ERROR IN Z AXIS (PID) ----------------
            
            # Safety check
            if current_depth > self.max_safe_depth:
                rospy.logwarn_throttle(1.0, "!!! SAFETY LIMIT REACHED (%.2fm) !!! Forcing ASCENT.", 
                                       current_depth)
                vz = -0.9 
                reached_z = False
                self.int_z = 0.0 
                e_z_or_depth = 0.0
            else:
                # Lógica normal de control
                e_depth = target_depth - current_depth
                e_z_or_depth = e_depth
                
                # PI D Control for Z
                vz_cmd = (self.kp_z * e_z_or_depth - self.kd_z * self.vz_est + self.int_z)
                
                if abs(vz_cmd) < self.vz_max - 1e-3:
                    self.int_z += self.ki_z * e_z_or_depth / self.rate_hz
                    self.int_z = self.clip(self.int_z, -self.int_z_limit, self.int_z_limit)
                
                vz = self.clip(vz_cmd, -self.vz_max, self.vz_max)
                reached_z = abs(e_z_or_depth) < self.tol_z
            
            # --- Waypoint Transition ---
            if reached_y and reached_z:
                rospy.loginfo("WP %d reached.", self.current_wp_index)
                self.current_wp_index += 1
                self.int_z = 0.0
                self.int_y = 0.0 
                vy, vz = 0.0, 0.0 
            
            # --- APPLY SMOOTHING (OR NOT) ---
            vy_out, vz_out = self.smooth(vy, vz)
            
            requester = "inspector_wp_{}".format(self.current_wp_index)
            self.publish_cmd(0.0, vy_out, vz_out, requester)
            
            # --- PUBLISHING FOR RQT_PLOT AND CURRENT TARGET ---
            # 1. Real Z
            self.pub_real_z.publish(Float64(current_depth))
            # 2. Target Z
            self.pub_target_z.publish(Float64(target_depth))
            # 3. Output Vz Command
            self.pub_vz_out.publish(Float64(vz_out*10))
            # 3b. Error in Z
            self.pub_error_z.publish(Float64(e_z_or_depth))

            # 4. CURRENT TARGET POINT
            self.publish_current_target_point(self.current_wp_index)
            
            rospy.loginfo_throttle(1.0, 
                "WP %d | ErrY:%.2f (IntY:%.2f) | ErrZ:%.2f | Cmd(vy:%.2f, vz:%.2f)",
                self.current_wp_index, e_y, self.int_y, e_z_or_depth, vy_out, vz_out)

        # --- State 3: FINISHED ---
        elif self.state == "FINISHED":
            vy, vz = self.smooth(0.0, 0.0)
            self.publish_cmd(0.0, vy, vz, "inspector_finished")
        else:
            vy, vz = self.smooth(0.0, 0.0)
            self.publish_cmd(0.0, vy, vz, "inspector_idle")


    def smooth(self, vy, vz):
        if not self.use_smoothing_flag:
            self.vy_f = vy
            self.vz_f = vz
            return vy, vz

        vy_f = self.alpha * vy + (1.0 - self.alpha) * self.vy_f
        vz_f = self.alpha * vz + (1.0 - self.alpha) * self.vz_f
        
        vy_f = self.clip(vy_f, self.vy_f - self.max_dv, self.vy_f + self.max_dv)
        vz_f = self.clip(vz_f, self.vz_f - self.max_dv, self.vz_f + self.max_dv)
        
        self.vy_f, self.vz_f = vy_f, vz_f
        return vy_f, vz_f

    def publish_cmd(self, vx, vy, vz, requester):
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