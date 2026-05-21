#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped, PointStamped
from net_hole_detector.msg import Detection3DArray  


class DetectionToWorldPose:
    def __init__(self):
        # -------- Params --------
        self.input_topic  = rospy.get_param("~input_topic",  "/net_hole_detector/detections_3d")
        self.output_topic = rospy.get_param("~output_topic", "/net_hole_detector/detections_pose_world")
        self.world_frame  = rospy.get_param("~world_frame", "world_ned")

        # ---------- Pose filtering ----------
        self.jump_threshold = rospy.get_param("~jump_threshold", 1.0)  # metros
        self.alpha = rospy.get_param("~alpha", 0.25)

        self.filtered_pose = None
        self.have_filtered_pose = False

        # -------- Publish rate --------
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)  # Hz
        self.last_pose = None

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_cb
        )

        # Score
        self.min_score    = rospy.get_param("~min_score", 0.0)

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
        self.w_area  = rospy.get_param("~w_area",  0.3)  # por defecto 0 => puro score

        # -------- TF2 --------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # -------- Pub/Sub --------
        self.pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, Detection3DArray, self.cb, queue_size=1)

        rospy.loginfo(
            "detection_to_world_pose: input=%s output=%s world_frame=%s min_score=%.3f area=[%.6f, %.6f] mode=%s",
            self.input_topic, self.output_topic, self.world_frame,
            self.min_score, self.min_area, self.max_area, self.selection_mode
        )

    def _get_area(self, det):
        # Tu msg parece tener width/height.
        # Si realmente es "length", cambia getattr(det,"height") por getattr(det,"length")
        w = float(getattr(det, "width", 0.0))
        h = float(getattr(det, "height", 0.0))
        return w * h
    
    def distance(self, p1, p2):
        dx = p1.pose.position.x - p2.pose.position.x
        dy = p1.pose.position.y - p2.pose.position.y
        dz = p1.pose.position.z - p2.pose.position.z
        return (dx*dx + dy*dy + dz*dz)**0.5
    def reject_jump(self, new_pose):
        if not self.have_filtered_pose:
            return True

        d = self.distance(new_pose, self.filtered_pose)

        if d > self.jump_threshold:
            rospy.logwarn_throttle(
                1.0,
                "Pose jump rejected: %.3f m > %.3f",
                d,
                self.jump_threshold
            )
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
    
    def timer_cb(self, event):
        if self.last_pose is not None:
            self.pub.publish(self.last_pose)

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

        # 4) Transformar a mundo
        try:
            p_world = self.tf_buffer.transform(p_cam, self.world_frame, timeout=rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF transform failed (%s -> %s): %s",
                                   msg.header.frame_id, self.world_frame, str(e))
            return

        # 5) Publicar pose
        pose = PoseStamped()
        pose.header.stamp = p_world.header.stamp
        pose.header.frame_id = self.world_frame

        pose.pose.position.x = p_world.point.x
        pose.pose.position.y = p_world.point.y
        pose.pose.position.z = p_world.point.z

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = -0.70710678
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.70710678

        # ----- Outlier rejection -----
        if not self.reject_jump(pose):
            return

        # ----- Low pass filtering -----
        filtered_pose = self.low_pass_filter(pose)

        self.last_pose = filtered_pose



if __name__ == "__main__":
    rospy.init_node("detection_to_world_pose")
    DetectionToWorldPose()
    rospy.spin()