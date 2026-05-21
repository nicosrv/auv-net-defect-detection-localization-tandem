#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros

import numpy as np
import tf.transformations as tf_trans

from geometry_msgs.msg import PoseStamped, PointStamped
from net_hole_detector.msg import Detection3DArray, CoordinatesError 


class DetectionToWorldPose:
    def __init__(self):
        # -------- Params --------
        self.input_topic  = rospy.get_param("~input_topic",  "/net_hole_detector/detections_3d")
        self.output_topic = rospy.get_param("~output_topic", "/net_hole_detector/hole")
        self.world_frame  = rospy.get_param("~world_frame", "world_ned")
        # To show the error between gorund truth and detection
        self.error_topic = rospy.get_param("~error_topic", "/net_hole_detector/error")

        # ---------- Pose filtering ----------
        self.jump_threshold = rospy.get_param("~jump_threshold", 10.0)  # metros
        self.alpha = rospy.get_param("~alpha", 0.5)

        self.filtered_pose = None
        self.have_filtered_pose = False

        # Score
        self.min_score    = rospy.get_param("~min_score", 0.4)

        # Área (width * height)
        # OJO: ajusta unidades según tu detector (m, px normalizado, etc.)
        self.min_area     = rospy.get_param("~min_area", 0.0)
        self.max_area     = rospy.get_param("~max_area", float("inf"))


        # Selección
        # "score" => elige el mayor score entre los que pasan filtros
        # "score_then_area" => score, y si empatan, mayor área
        # "score_area_combo" => combina score y área (normalizada) con pesos
        self.selection_mode = rospy.get_param("~selection_mode", "score_area_combo")
        # Para score_area_combo:
        self.w_score = rospy.get_param("~w_score", 0.7)
        self.w_area  = rospy.get_param("~w_area",  0.3) 

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # -------- Pub/Sub --------
        self.pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, Detection3DArray, self.cb, queue_size=1)
    
        # -------- Error Publisher --------
        self.err_pub = rospy.Publisher(self.error_topic, CoordinatesError, queue_size=10)

        rospy.loginfo(
            "detection_to_world_pose ready! input=%s world_frame=%s",
            self.input_topic, self.world_frame
        )

        # -------- Publish rate --------
        # self.publish_rate = rospy.get_param("~publish_rate", 10.0)  # Hz
        # self.last_pose = None

        # self.timer = rospy.Timer(
        #     rospy.Duration(1.0 / self.publish_rate),
        #     self.timer_cb
        # )

    def _get_area(self, det):
        # Tu msg parece tener width/height.
        # Si realmente es "length", cambia getattr(det,"height") por getattr(det,"length")
        w = float(getattr(det, "width", 0.0))
        h = float(getattr(det, "height", 0.0))
        return w * h
    
    def distance(self, p1, p2): # esto me devuelve la distancia euclidea en 3d entre 2 puntos
        dx = p1.pose.position.x - p2.pose.position.x
        dy = p1.pose.position.y - p2.pose.position.y
        dz = p1.pose.position.z - p2.pose.position.z
        return (dx*dx + dy*dy + dz*dz)**0.5 

    def reject_jump(self, new_pose):
        if not self.have_filtered_pose:
            return True

        d = self.distance(new_pose, self.filtered_pose)

        if d > self.jump_threshold:
            rospy.logwarn_throttle(1.0, "Pose jump rejected! %.2f m (Threshold: %.2f)", d, self.jump_threshold)
            return False

        return True
    
    def low_pass_filter(self, new_pose):
        if not self.have_filtered_pose:
            self.filtered_pose = new_pose
            self.have_filtered_pose = True
            return new_pose

        a = self.alpha
        fp = self.filtered_pose.pose.position
        np = new_pose.pose.position

        fp.x = a*np.x + (1-a)*fp.x
        fp.y = a*np.y + (1-a)*fp.y
        fp.z = a*np.z + (1-a)*fp.z

        self.filtered_pose.header = new_pose.header

        return self.filtered_pose
    
    # def timer_cb(self, event):
    #     if self.last_pose is not None:
    #         self.pub.publish(self.last_pose)

    def _passes_filters(self, det):
        score = float(getattr(det, "score", 0.0))
        if score < self.min_score:
            return False

        area = self._get_area(det)
        if area < self.min_area or area > self.max_area:
            return False

        return True

    def _select_detection(self, detections):
        """
        Selecciona una detección entre las filtradas.
        """
        if self.selection_mode == "score":
            return max(detections, key=lambda d: float(getattr(d, "score", 0.0)))

        if self.selection_mode == "score_then_area":
            return max(detections, key=lambda d: (float(getattr(d, "score", 0.0)), self._get_area(d)))

        if self.selection_mode == "score_area_combo":
            # Normalización simple del área para no reventar escalas:
            # (si max_area es infinito, usa max área observada en este frame)
            areas = [self._get_area(d) for d in detections]
            a_max = max(areas) if areas else 1.0
            a_max = a_max if a_max > 1e-12 else 1.0

            def key(d):
                s = float(getattr(d, "score", 0.0))
                a = self._get_area(d) / a_max
                return self.w_score * s + self.w_area * a

            return max(detections, key=key)

        # Fallback
        return max(detections, key=lambda d: float(getattr(d, "score", 0.0)))

    def cb(self, msg):
        if not hasattr(msg, "detections") or len(msg.detections) == 0:
            return

        # 1) Filtrar por score y área
        filtered = [d for d in msg.detections if self._passes_filters(d)]
        if not filtered:
            return

        # 2) Elegir mejor (por score, y opcionalmente área)
        det = self._select_detection(filtered)

        # 3) Punto en frame local
        p_cam = PointStamped()
        p_cam.header = msg.header
        p_cam.point.x = det.x
        p_cam.point.y = det.y
        p_cam.point.z = det.z

        # 4) Transformar a mundo (usando numpy)
        try:
            trans = self.tf_buffer.lookup_transform(
                self.world_frame, 
                msg.header.frame_id, 
                msg.header.stamp, 
                rospy.Duration(0.1)
            )
            
            # --- MATEMÁTICAS 3D MANUALES ---
            t = trans.transform.translation
            r = trans.transform.rotation
            
            # Crear matriz de transformación 4x4 a partir del cuaternión
            mat = tf_trans.quaternion_matrix([r.x, r.y, r.z, r.w])
            
            # Añadir la traslación a la matriz
            mat[0, 3] = t.x
            mat[1, 3] = t.y
            mat[2, 3] = t.z
            
            # Convertir nuestro punto a vector de 4 elementos [x, y, z, 1]
            p_vec = np.array([p_cam.point.x, p_cam.point.y, p_cam.point.z, 1.0])
            
            # Multiplicar matriz por vector para obtener el punto en el mundo
            p_world_vec = np.dot(mat, p_vec)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF Wait failed: %s", str(e))
            return
        
        # 4.5) Listener to fixed tf (ground truth)
        try:
            gt_trans = self.tf_buffer.lookup_transform(
                self.world_frame,
                "hole_ground_truth",
                rospy.Time(0),
                rospy.Duration(0.1)
            )

            t_gt = gt_trans.transform.translation

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF Wait failed: %s", str(e))
            return

        # 5) Crear Pose final
        pose = PoseStamped()
        pose.header.stamp = trans.header.stamp
        pose.header.frame_id = self.world_frame

        # Asignar los nuevos valores calculados
        pose.pose.position.x = p_world_vec[0]
        pose.pose.position.y = p_world_vec[1]
        pose.pose.position.z = p_world_vec[2]

        # orientación fija (provisional)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = -0.70710678
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.70710678

        # 6) Noise filtering
        # ----- Outlier rejection -----
        if not self.reject_jump(pose):
            return

        # ----- Low pass filtering -----
        filtered_pose = self.low_pass_filter(pose)

        # 7) Publish immediately
        self.pub.publish(filtered_pose)

        # 8) Crear Mensaje de Errores final
        c_err = CoordinatesError()
        c_err.header.stamp = trans.header.stamp
        c_err.header.frame_id = self.world_frame

        # Asignar los valores
        c_err.x_err = abs(filtered_pose.pose.position.x - t_gt.x)
        c_err.y_err = abs(filtered_pose.pose.position.y - t_gt.y)
        c_err.z_err = abs(filtered_pose.pose.position.z - t_gt.z)

        c_err.total_error = ((c_err.x_err)**2 + (c_err.y_err)**2 + (c_err.z_err)**2)**(1/2)

        # Publicar el mensaje
        self.err_pub.publish(c_err)


if __name__ == "__main__":
    rospy.init_node("detection_to_world_pose")
    DetectionToWorldPose()
    rospy.spin()