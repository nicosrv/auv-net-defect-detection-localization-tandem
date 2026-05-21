#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from math import sin, cos, sqrt, pi
from net_hole_detector.msg import AxisPoints, NetStats
import yaml
import os

class ImageGeolocalizationNode:

    def __init__(self):
        rospy.init_node('image_geolocalization_node', anonymous=True)

        # Parameters (can be set via rosparam)
        self.image_topic = rospy.get_param("~masked_image_topic", "/net_hole_detector/net_mask_yolo_fused")
        self.info_topic = rospy.get_param("holes_info_topic", "/net_hole_detector/net_stats")
        self.output_topic = "/net_hole_detector/blob_axis_orientation"
        self.frame_id = "frame_id"

        # Params
        default_yaml = rospy.get_param("/net_hole_detector_3d/calibration_file", "")
        print('yaml path = ' + str(default_yaml))
        self.__load_from_yaml_file(default_yaml)

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1
        )

        self.image_info_sub = rospy.Subscriber(
            self.info_topic,
            NetStats,
            self.info_callback,
            queue_size=1
        )
 
        # Publisher
        self.pose_pub = rospy.Publisher(
            self.output_topic,
            AxisPoints,
            queue_size=10
        )

        rospy.loginfo("Image Geolocalization Node Initialized")
        rospy.spin()

    def info_callback(self, msg):
        self.__scale_m_per_px = msg.scale_m_per_px
        self.__median_hole_area_px = msg.median_hole_area_px
        self.__median_hole_area_m2 = msg.median_hole_area_m2

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        self.process_image(cv_image)

    """
    Function: load_from_yaml_file

    """
    def __load_from_yaml_file(self, file_path):
        """
        Reads an standard ROS calibration YAML file (camera_calibration)
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] Config file not found: {file_path}")
            return

        print(f"[CameraGeometry] Loading config from: {file_path}")

        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f)

                # 1. Resolution
                self.img_w = data.get('image_width', 1280)
                self.img_h = data.get('image_height', 720)

                # 2. Intrinsec Matrix (K) -> For Raw
                k_data = data['camera_matrix']['data']
                self.k_matrix = np.array(k_data).reshape(3, 3)

                # 3. Distorsion Coefficients (D)
                d_data = data['distortion_coefficients']['data']
                self.dist_coeffs = np.array(d_data)

                # 4. Projection Matrix (P) -> For RECT
                # Sometimes .yaml has P matrix and sometimes it has not, we try to load it
                if 'projection_matrix' in data:
                    p_data = data['projection_matrix']['data']
                    # P is 3x4, we stay with 3x3
                    self.p_matrix = np.array(p_data).reshape(3, 4)[:3, :3]
                else:
                    # If there is not P, we use K as a fallback to avoid crashing
                    print("[CameraGeometry] WARNING: 'projection_matrix' not found in YAML. Using K as fallback.")
                    self.p_matrix = self.k_matrix

                # Initialize with K by default (Raw mode)
                self.camera_matrix = self.k_matrix
                self.fx = self.k_matrix[0, 0]
                self.fy = self.k_matrix[1, 1]
                self.cx = self.k_matrix[0, 2]
                self.cy = self.k_matrix[1, 2]

                self.is_calibrated = True
                print(f" -> YAML Calibration OK. FX: {self.fx:.2f}, FY: {self.fy:.2f}")

            except Exception as e:
                print(f"[ERROR] Failed to parse YAML. Check if it is standard ROS format. Error: {e}")


    def process_image(self, masked_image):

        # =========================
        # 1. Convertir a Gris
        # =========================
        # print(masked_image.shape)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        _, mask_blobs = cv2.threshold(masked_image, 1, 255, cv2.THRESH_BINARY)

        # =========================
        # 2. Morfología: eliminar ruido y deformaciones
        # =========================
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Closing = dilatación seguida de erosión
        mask_clean = cv2.morphologyEx(mask_blobs, cv2.MORPH_CLOSE, kernel)

        # (opcional) Opening adicional para ruido pequeño
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        # print('This is mask_clean: shape = ' + str(mask_clean.shape) + '. Type = ' + str(type(mask_clean)) + '. Unique = ' + str(np.unique(mask_clean * 255)))

        # =========================
        # 3. Encontrar contornos
        # =========================
        # mask_clean = cv2.cvtColor(mask_clean, cv2.COLOR_BGR2GRAY)
        contours, hierarchi = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # # Tomar el blob más grande (típico en segmentación semántica)
        # largest_contour = max(contours, key=cv2.contourArea)
        # print('This is contour = ' + str(len(contours)))
        # print('This is largest = ' + str(len(largest_contour)))

        # =========================
        # 4. Ajustar elipse y calcular ejes
        # =========================
        vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        meters_per_pixel = 0.00067 # TODO: Llegir dels càlculs de Alejandro
        area_xarxa = 100 # TODO: Llegir dels càlculs de Alejandro
        i = 0
        for contour in contours:
            if len(contour) < 5:
                print('Contorn molt petit!')
                continue

            ellipse = cv2.fitEllipse(contour)

            (center_x, center_y), (width, height), angle = ellipse
            # print('Center x = ' + str(center_x) + '. Center y = ' + str(center_y))
            # print('Width = ' + str(width))
            # print('Height = ' + str(height))

            eje_mayor = max(width, height)
            eje_menor = min(width, height)

            # print(f"Eje mayor: {eje_mayor:.2f} px")
            # print(f"Eje menor: {eje_menor:.2f} px")
            # print(f"Ángulo: {angle:.2f} grados")

            area = pi * width/2 + height/2
            # print('This is area = ' + str(area) + '. Of blop = ' + str(i))
            if area < self.__median_hole_area_px: # Forat més petit que els foradets de la xarxa -> Descartat!
                # print('Area massa petita!: area = ' + str(area) + '. threshold = ' + str(self.__median_hole_area_px))
                continue

            cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)


            # =========================
            # 5. Visualizació amb matplotlib
            # =========================

            scale_m_per_px = self.__scale_m_per_px

            x_dis = np.cos(angle*pi/180 + pi/2) * height / 2
            y_dis = np.sin(angle*pi/180 + pi/2) * height / 2
            x_gran_borders = [center_x-x_dis, center_x+x_dis]
            y_gran_borders = [center_y-y_dis, center_y+y_dis]
            eix_gran_dist = sqrt((x_gran_borders[0] - x_gran_borders[1])**2 + (y_gran_borders[0] - y_gran_borders[1])**2)

            x_dis = np.cos(angle*pi/180) * width / 2
            y_dis = np.sin(angle*pi/180) * width / 2
            x_petit_borders = [center_x-x_dis, center_x+x_dis]
            y_petit_borders = [center_y-y_dis, center_y+y_dis]
            eix_petit_dist = sqrt((x_petit_borders[0] - x_petit_borders[1])**2 + (y_petit_borders[0] - y_petit_borders[1])**2)

            # Ara el 0,0 de la imatge està en un cantó. A continuació per a passar el 0,0 al punt de la lent
            x_gran_borders = [(center_x-x_dis - self.cx) * scale_m_per_px, (center_x+x_dis - self.cx) * scale_m_per_px]
            y_gran_borders = [(center_y-y_dis - self.cy) * scale_m_per_px, (center_y+y_dis - self.cy) * scale_m_per_px]
            eix_gran_dist = sqrt((x_gran_borders[0] - x_gran_borders[1])**2 + (y_gran_borders[0] - y_gran_borders[1])**2)

            x_petit_borders = [(center_x-x_dis - self.cx) * scale_m_per_px, (center_x+x_dis - self.cx) * scale_m_per_px]
            y_petit_borders = [(center_y-y_dis - self.cy) * scale_m_per_px, (center_y+y_dis - self.cy) * scale_m_per_px]
            eix_petit_dist = sqrt((x_petit_borders[0] - x_petit_borders[1])**2 + (y_petit_borders[0] - y_petit_borders[1])**2)


            # =========================
            # 6. Crear missatge i publicar
            # =========================        
            axis_points = AxisPoints()

            axis_points.header.stamp = rospy.Time.now()
            axis_points.header.frame_id = self.frame_id

            axis_points.major_axis_p1.x = x_gran_borders[0]
            axis_points.major_axis_p2.x = x_gran_borders[1]
            axis_points.major_axis_p1.y = y_gran_borders[0]
            axis_points.major_axis_p2.y = y_gran_borders[1]

            axis_points.minor_axis_p1.x = x_petit_borders[0]
            axis_points.minor_axis_p2.x = x_petit_borders[1]
            axis_points.minor_axis_p1.y = y_petit_borders[0]
            axis_points.minor_axis_p2.y = y_petit_borders[1]

            axis_points.orientation = angle
            axis_points.major_axis_len = eix_gran_dist
            axis_points.minor_axis_len = eix_petit_dist

            # Publish
            self.pose_pub.publish(axis_points)


    def yaw_to_quaternion(self, yaw):
        """
        Converts yaw angle (in radians) into quaternion.
        Assumes roll = pitch = 0.
        """
        qx = 0.0
        qy = 0.0
        qz = sin(yaw / 2.0)
        qw = cos(yaw / 2.0)
        return [qx, qy, qz, qw]


if __name__ == "__main__":
    try:
        ImageGeolocalizationNode()
    except rospy.ROSInterruptException:
        pass