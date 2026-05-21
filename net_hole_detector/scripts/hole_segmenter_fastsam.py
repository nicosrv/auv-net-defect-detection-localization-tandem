#!/usr/bin/env python3
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ultralytics import FastSAM
from net_hole_detector.msg import BoundingBoxArray


class HoleSegmenterFastSAM:
    """
    Nodo ROS que recibe:
      - imagen de cámara
      - bounding boxes de YOLO

    Y publica:
      - máscara binaria del agujero
      - imagen debug con bbox + máscara

    Idea:
      YOLO localiza la zona aproximada del agujero.
      FastSAM segmenta la imagen.
      Se selecciona la máscara de FastSAM que más solapa con la bbox de YOLO.
    """

    def __init__(self):
        rospy.init_node("hole_segmenter_fastsam")

        # -------- Parámetros --------
        self.image_topic = rospy.get_param(
            "~image_topic",
            "/girona500/xiroi/stereo_ch3/left_optical/image_color"
        )

        self.bbox_topic = rospy.get_param(
            "~bbox_topic",
            "/net_hole_detector/stereo_left/bounding_boxes"
        )

        self.mask_topic = rospy.get_param(
            "~mask_topic",
            "/net_hole_detector/stereo_left/hole_mask"
        )

        self.debug_topic = rospy.get_param(
            "~debug_topic",
            "/net_hole_detector/stereo_left/hole_mask_debug"
        )

        self.weights_path = rospy.get_param(
            "~fastsam_weights",
            "/home/rosuser/catkin_ws/src/repo_ros/net_hole_detector/weights/FastSAM-s.pt"
        )

        self.conf = float(rospy.get_param("~conf", 0.4))
        self.iou = float(rospy.get_param("~iou", 0.9))
        self.imgsz = int(rospy.get_param("~imgsz", 640))
        self.period = float(rospy.get_param("~period", 1.0))

        self.last_process_time = rospy.Time(0)

        self.bridge = CvBridge()

        rospy.loginfo("[FastSAM] Cargando modelo desde: %s", self.weights_path)
        self.model = FastSAM(self.weights_path)
        rospy.loginfo("[FastSAM] Modelo cargado correctamente.")

        # -------- Subscriptores sincronizados --------
        sub_img = Subscriber(self.image_topic, Image)
        sub_box = Subscriber(self.bbox_topic, BoundingBoxArray)

        self.sync = ApproximateTimeSynchronizer(
            [sub_img, sub_box],
            queue_size=10,
            slop=0.15
        )
        self.sync.registerCallback(self.callback)

        # -------- Publicadores --------
        self.pub_mask = rospy.Publisher(
            self.mask_topic,
            Image,
            queue_size=1
        )

        self.pub_debug = rospy.Publisher(
            self.debug_topic,
            Image,
            queue_size=1
        )

        rospy.loginfo("[FastSAM] Escuchando imagen: %s", self.image_topic)
        rospy.loginfo("[FastSAM] Escuchando bboxes: %s", self.bbox_topic)
        rospy.loginfo("[FastSAM] Publicando máscara en: %s", self.mask_topic)
        rospy.loginfo("[FastSAM] Publicando debug en: %s", self.debug_topic)

    def callback(self, img_msg, bbox_msg):
        now = rospy.Time.now()

        if (now - self.last_process_time).to_sec() < self.period:
            return

        self.last_process_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            height, width = cv_img.shape[:2]

            final_mask = np.zeros((height, width), dtype=np.uint8)
            debug_img = cv_img.copy()

            # Si YOLO no detecta nada, publicamos máscara vacía.
            if not bbox_msg.boxes:
                mask_msg = self.bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
                mask_msg.header = img_msg.header
                self.pub_mask.publish(mask_msg)

                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                debug_msg.header = img_msg.header
                self.pub_debug.publish(debug_msg)

                rospy.loginfo_throttle(2.0, "[FastSAM] Sin bboxes de YOLO.")
                return

            # Ejecutar FastSAM sobre la imagen completa
            results = self.model(
                cv_img,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                retina_masks=True
            )

            result = results[0]

            if result.masks is None:
                rospy.logwarn("[FastSAM] No se han generado máscaras.")
                return

            masks = result.masks.data.cpu().numpy()

            # Asegurar que las máscaras tienen el tamaño de la imagen original
            resized_masks = []
            for m in masks:
                m_uint8 = (m > 0.5).astype(np.uint8) * 255
                if m_uint8.shape[0] != height or m_uint8.shape[1] != width:
                    m_uint8 = cv2.resize(m_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(m_uint8)

            # Para cada bbox de YOLO, buscamos la máscara de FastSAM que más solapa
            for bb in bbox_msg.boxes:
                u1 = int((bb.x - bb.w / 2.0) * width)
                u2 = int((bb.x + bb.w / 2.0) * width)
                v1 = int((bb.y - bb.h / 2.0) * height)
                v2 = int((bb.y + bb.h / 2.0) * height)

                u1 = max(0, min(width - 1, u1))
                u2 = max(0, min(width - 1, u2))
                v1 = max(0, min(height - 1, v1))
                v2 = max(0, min(height - 1, v2))

                if u2 <= u1 or v2 <= v1:
                    continue

                bbox_mask = np.zeros((height, width), dtype=np.uint8)
                bbox_mask[v1:v2, u1:u2] = 255

                best_mask = None
                best_score = 0.0

                for m in resized_masks:
                    intersection = np.logical_and(m > 0, bbox_mask > 0).sum()
                    mask_area = (m > 0).sum()
                    bbox_area = (bbox_mask > 0).sum()

                    if mask_area == 0 or bbox_area == 0:
                        continue

                    # Score: cuánto de la máscara cae dentro de la bbox
                    score = intersection / float(mask_area)

                    if score > best_score:
                        best_score = score
                        best_mask = m

                # Si encuentra una máscara razonable, la añadimos a la máscara final
                if best_mask is not None and best_score > 0.05:
                    final_mask = cv2.bitwise_or(final_mask, best_mask)

                    # Dibujar bbox
                    cv2.rectangle(debug_img, (u1, v1), (u2, v2), (0, 255, 255), 2)

                    # Pintar máscara en verde
                    green_overlay = np.zeros_like(debug_img)
                    green_overlay[:, :, 1] = final_mask
                    debug_img = cv2.addWeighted(debug_img, 1.0, green_overlay, 0.4, 0)

                    rospy.loginfo(
                        "[FastSAM] Máscara seleccionada | bbox score=%.2f | YOLO score=%.2f",
                        best_score,
                        bb.score
                    )
                else:
                    rospy.logwarn("[FastSAM] No se encontró máscara útil para una bbox.")

            # Publicar máscara final
            mask_msg = self.bridge.cv2_to_imgmsg(final_mask, encoding="mono8")
            mask_msg.header = img_msg.header
            self.pub_mask.publish(mask_msg)

            # Publicar imagen debug
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = img_msg.header
            self.pub_debug.publish(debug_msg)

        except Exception as e:
            rospy.logerr("[FastSAM] Error: %s", str(e))


if __name__ == "__main__":
    try:
        HoleSegmenterFastSAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

