#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from cola2_msgs.msg import BodyVelocityReq, NavSts

class InspectorAbsoluto(object):
    def __init__(self):
        rospy.init_node("inspector_absoluto")

        # --- OBJETIVO ---
        self.target_x = 1.5   
        self.target_y = -0.5  
        self.target_z = 1.0   
        self.target_yaw = math.radians(-90.0) 

        # --- CONTROLADORES ---
        self.kp_pos = 0.15
        self.kp_yaw = 0.12 # Un poco más firme para el final
        
        # --- UMBRALES ---
        self.pos_arrival_threshold = 0.15 # 15cm para considerar que ha llegado
        self.yaw_threshold = math.radians(2.0)

        # --- ESTADO ---
        self.arrived_at_pos = False

        self.pub_vel = rospy.Publisher("/girona500/controller/body_velocity_req", BodyVelocityReq, queue_size=1)
        self.sub_nav = rospy.Subscriber("/girona500/navigator/navigation", NavSts, self.nav_callback)

        rospy.loginfo("Inspector v3: Estrategia secuencial (Posición -> luego Orientación)")

    def nav_callback(self, msg):
        # 1. Posición actual
        current_x = msg.position.north
        current_y = msg.position.east
        current_z = msg.position.depth
        current_yaw = msg.orientation.yaw

        # 2. Errores Globales
        ex_g = self.target_x - current_x
        ey_g = self.target_y - current_y
        ez = self.target_z - current_z
        
        dist_euclidea = math.sqrt(ex_g**2 + ey_g**2)
        
        error_yaw = self.target_yaw - current_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

        # 3. Lógica de estados: ¿Hemos llegado ya a la posición?
        if dist_euclidea < self.pos_arrival_threshold:
            self.arrived_at_pos = True
        
        # 4. Cálculo de comandos
        vx_cmd, vy_cmd, vyaw_cmd = 0.0, 0.0, 0.0

        if not self.arrived_at_pos:
            # ESTADO 1: Ir a la posición (manteniendo yaw actual o fijo)
            # Transformamos error global a body frame
            vx_cmd = (ex_g * math.cos(current_yaw) + ey_g * math.sin(current_yaw)) * self.kp_pos
            vy_cmd = (-ex_g * math.sin(current_yaw) + ey_g * math.cos(current_yaw)) * self.kp_pos
            vyaw_cmd = 0.0 # No giramos mientras nos movemos para no marear al navegador
        else:
            # ESTADO 2: Ya estamos en el sitio, ahora rotamos
            vx_cmd = 0.0 
            vy_cmd = 0.0
            if abs(error_yaw) > self.yaw_threshold:
                vyaw_cmd = self.kp_yaw * error_yaw
            else:
                vyaw_cmd = 0.0
                rospy.loginfo_once("!!! OBJETIVO ALCANZADO Y ORIENTADO !!!")

        # El eje Z (profundidad) siempre activo para no hundirse
        vz_cmd = ez * self.kp_pos

        # 5. Publicar (con límites integrados)
        self.publish_velocity(
            self.clamp(vx_cmd, 0.2), 
            self.clamp(vy_cmd, 0.2), 
            self.clamp(vz_cmd, 0.2), 
            self.clamp(vyaw_cmd, 0.15)
        )
        
        mode = "ROTANDO" if self.arrived_at_pos else "VIAJANDO"
        rospy.loginfo_throttle(2.0, "[%s] Dist: %.2fm | ErrYaw: %.2f deg", mode, dist_euclidea, math.degrees(error_yaw))

    def clamp(self, val, limit):
        return max(-limit, min(limit, val))

    def publish_velocity(self, vx, vy, vz, vyaw):
        req = BodyVelocityReq()
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = "girona500/base_link"
        req.goal.priority = 50 
        req.twist.linear.x, req.twist.linear.y, req.twist.linear.z = vx, vy, vz
        req.twist.angular.z = vyaw
        req.disable_axis.roll, req.disable_axis.pitch = True, True
        self.pub_vel.publish(req)

if __name__ == '__main__':
    try:
        InspectorAbsoluto()
        rospy.spin()
    except rospy.ROSInterruptException: pass