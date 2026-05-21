#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

# ----- ROS MESSAGES -----
from sensor_msgs.msg import Image, CameraInfo
from net_hole_detector.srv import Trigger, TriggerResponse

# ----- CUSTOM MESSAGES -----
from net_hole_detector.msg import BoundingBox, BoundingBoxArray
from net_hole_detector.msg import Detection3D, Detection3DArray
from net_hole_detector.msg import NetStats

# ----- OWN CLASSES -----
from net_hole_detector.camera_geometry import CameraGeometry
from net_hole_detector.scale_estimator import ScaleEstimator

MAX_DETECTIONS = 5
MAX_BB_AREA_PERC = 0.25

class NetHoleDetectorNode:
    def __init__(self):
        rospy.init_node('net_hole_detector')

        # Topics name gathering
        self.image_topic = rospy.get_param("~image_topic", "/girona500/bravo/gripper/camera/image_rect/compressed")
        self.info_topic = rospy.get_param("~camera_info_topic", "/girona500/bravo/gripper/camera/camera_info")
        self.yolo_topic = rospy.get_param("~yolo_topic", "/yolo/detections")
        
        # We try to load a Calibration YAML file (In case CameraInfo topic does not exist)
        default_yaml = rospy.get_param("~calibration_file", "") 
        
        # Frame ID (In which coordinates system is the point)
        self.camera_frame = rospy.get_param("~camera_frame", "bravo_camera_optical_frame")
        
        # Get specific parameter to know if the image is rectified or not.
        self.is_already_rectified = rospy.get_param("~is_rectified", False)

        if self.is_already_rectified:
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RECTIFIED. Mode: PASSTHROUGH.")
        else:
            rospy.loginfo(f"[Node] Topic '{self.image_topic}' detected as RAW. Mode: UNDISTORT active.")

        # CameraGeometry: Tries to parser a YAML if it exist, otherwise, it initializes empty
        self.geo = CameraGeometry(yaml_path=default_yaml if default_yaml else None)
        if self.geo.is_calibrated: # Once the YAML is loaded, we force the Right Matrix Selection
            self.geo.select_matrix(self.is_already_rectified)
        
        # Estimator: Blobs Logic to get Z
        self.estimator = ScaleEstimator() # TODO: implementar mecanisme per a triar com calcular la Z

        # OpenCV-ROS Bridge
        self.bridge = CvBridge()

        # Variables initialization
        self.latest_z = None # Store the "Z" calculated by the last image
        self.last_z_time = rospy.Time(0) # Initialize with time 0 so the first check fails until we get a real image
        self.current_image = None
        
        # Subscriptions
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")

        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(f"[Node] Listening to Images in: {self.image_topic}")

        self.sub_yolo = rospy.Subscriber(self.yolo_topic, BoundingBoxArray, self.yolo_callback)
        rospy.loginfo(f"[Node] Listening to YOLO in: {self.yolo_topic}")


        # Initialize publishers
        self.pub_point = rospy.Publisher("net_hole_detector/detections_3d", Detection3DArray, queue_size=1)
        self.pub_debug_img = rospy.Publisher('/net_hole_detector/debug_image', Image, queue_size=1)
        # Publisher of the Mask that fuses blobs and YOLO bboxes
        self.mask_yolo_pub = rospy.Publisher('/net_hole_detector/net_mask_yolo_fused', Image, queue_size=1)
        # Publisher with Median Area and Scale of every image
        self.net_stats_pub = rospy.Publisher('/net_hole_detector/net_stats', NetStats, queue_size=1)

        # Initialize services
        is_new_camera_selected_service = rospy.ServiceProxy('net_hole_detector/update_camera_info_srv', Trigger, self.__camera_change_callback)

    """
    Function: camera_change_callback

    """
    def __camera_change_callback(self, req):
        self.sub_info.unregister() 
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback)
        rospy.loginfo(f"[Node] Listening to CameraInfo in: {self.info_topic}")
        
        response = TriggerResponse()
        response.success = True
        response.message = "Request processed!"
        return response 

    """
    Function: info_callback
    
    This will be executed once the bagfile sends a calibration message.
    Overwrittes any parsed YAML
    """
    def info_callback(self, msg):
        try:
            self.geo.set_camera_info(msg)

            # Every Time new info arrives, we ensure fx, fy, cx, cy actualizes according to our mode (Rect vs Raw)
            self.geo.select_matrix(self.is_already_rectified)

            self.sub_info.unregister() 
            rospy.loginfo("Calibración recibida y guardada. Desuscribiendo del topic de info.")
        except Exception as e:
            rospy.logerr("Error - {e}")

    """
    Function: image_callback

    """
    def image_callback(self, msg):
        # 1. Decode Image
        try:
            if hasattr(msg, 'format') and "compressed" in str(type(msg)):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error decodificando imagen: {e}")
            return
        
        # SAFETY CHECK: If decoding returns None (corrupted data), we stop
        if cv_img is None:
            return

        # # 3. Smart Pre-Processing (Undistort)
        img_process = cv_img

        # 4. Scale Estimator
        # Detection blobs, area calculation... logic
        scale, median_area_px, _, _, _ = self.estimator.get_scale_and_images(img_process, real_area_m2=(0.015*0.015), yolo_bboxes=[])
        
        if scale:
            # ¡SUCCESS! We store (update) Z and the time in the Memory of the class
            self.latest_z = self.geo.get_z_distance(scale)
            self.last_z_time = rospy.Time.now()
            # rospy.loginfo(f"Estimated distance: {z:.2f} m")

            # --- PUBLISH NET STATS ---
            stats_msg = NetStats()
            stats_msg.header = msg.header
            stats_msg.scale_m_per_px = float(scale)
            stats_msg.median_hole_area_px = float(median_area_px)
            stats_msg.median_hole_area_m2 = float(median_area_px * (scale*scale)) # 2 dimensions not 1

            self.net_stats_pub.publish(stats_msg)
        else:
            # If we do not see the net, we preserve the last known Z
            pass

        # Store current image in memory
        self.current_image = img_process.copy()


    # =========================================================
    # CALLBACK 3: PROCESS YOLO -> CALCULATE 'X, Y' AND PUBLISH
    # =========================================================
    """
    Function: yolo_callback
    """
    def yolo_callback(self, msg):
        """
        It executes everytime YOLO sees something
        """
        # ===========================
        # 1. SANITY CHECKS (Safety)
        # ===========================
        if len(msg.boxes) == 0:
            return
        
        if self.latest_z is None:
            rospy.logwarn_throttle(2, "[Yolo] Object detected, but Z distance unkwown (waiting for image...)")
            return
        
        if self.current_image is None:
            return
        
        # Check Timeout (Caducity)
        time_diff = rospy.Time.now() - self.last_z_time
        # If it is 0.5s or older, its dangerous
        if time_diff.to_sec() > 0.5:
            rospy.logwarn_throttle(2, f"[Yolo] Z data too old ({time_diff.to_sec():.2f}s). Ignoring Yolo to avoid mistakes.")
            return
        
        # Calibration
        if not self.geo.is_calibrated:
            rospy.logwarn_throttle(2, "[Yolo] Unable to get Camera calibration. Impossible to project 3D")
            return
        
        # ===========================
        # 2. DATA PREPARATION
        # ===========================
        # Prepare image to draw (if it exists). We make a copy to not damage the original one
        debug_img = self.current_image.copy()

        # 2. Prepare boxes for the estimator. Needed format [(x,y,w,h), ...]
        boxes_pixels_for_estimator = []

        # "TOP-K" FILTER: Sort and Cut
        # a) Sort all bboxes by score,so we stay with the 'good' ones.
        all_boxes_sorted = sorted(msg.boxes, key=lambda b: b.score, reverse=True)
        # b) Aply the cut.
        best_boxes = all_boxes_sorted[:MAX_DETECTIONS]

        # Prepare output message (3D Detections)
        out_msg = Detection3DArray()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = self.camera_frame
        out_msg.detections = []

        # ===========================
        # 3. MAIN LOOP (Process every box)
        # ===========================
        # 5. Iterate over BEST bboxes:
        for box in best_boxes:
            # box is and object (pose, dimensions, ...)
            # We expect a list: [label, x, y, w, h]

            # Extract data from ROS message
            bbox_list = [
                box.class_id,  # Posición 0
                box.x,         # Posición 1 (x_n)
                box.y,         # Posición 2 (y_n)
                box.w,         # Posición 3 (w_n)
                box.h          # Posición 4 (h_n)
            ]

            # ===========================
            # BBOX SIZE FILTER: MAXIMUM AREA 
            # ===========================
            area_ratio = box.w * box.h

            if area_ratio > MAX_BB_AREA_PERC:
                rospy.logdebug(f"Discarding huge bbox: {area_ratio*100:.1f} of the area")

                # We try to raw it in RED for the DEBUG
                try:
                    x1, y1, x2, y2 = self.geo.get_bbox_corners_pixels(bbox_list)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    texto_debug = f"Filtro: {area_ratio*100:.1f}%"
                    cv2.putText(debug_img, texto_debug, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception as e:
                    pass

                # Now we forget this bbox, we go to the next one
                continue
            
            try:
                # Transform 0-1 -> Pixels (u, v)
                u, v = self.geo.yolo_to_pixels(bbox_list)

                # Project from 2D -> 3D (X, Y, Z)
                point_3d = self.geo.project_pixel_to_3d(u, v, self.latest_z)

                # Calculate Real Width and Height
                real_w, real_h = self.geo.get_object_dimensions(box.w, box.h, self.latest_z)
                
                # --- PREPARE DATA FOR ESTIMATOR & VISUALIZATION ---
                x1, y1, x2, y2 = self.geo.get_bbox_corners_pixels(bbox_list)

                # Height and Width with pixels
                w_px = x2 - x1
                h_px = y2 - y1

                # Format: (x_top_left, y_top_left, width, height)
                boxes_pixels_for_estimator.append((x1, y1, w_px, h_px))

                # --- CREATE ROS MESSAGE ---

                if hasattr(point_3d, '__len__'):
                    # Create individual object
                    det_3d = Detection3D()
                    det_3d.class_id = box.class_id
                    det_3d.score = box.score
                    det_3d.x = point_3d[0]
                    det_3d.y = point_3d[1]
                    det_3d.z = point_3d[2]
                    det_3d.width = real_w
                    det_3d.height = real_h

                    # Add to the list
                    out_msg.detections.append(det_3d)

                    # DRAW IN IMAGE (if we have an image)
                    # Draw Rectangle
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Prepare text (Z and Width)
                    label = f"Z:{det_3d.z:.2f}m W:{real_w:.2f}m"
                    
                    # Fondo negro para el texto (para leerlo bien)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Dibujamos cajita negra encima del rectángulo
                    cv2.rectangle(debug_img, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
                    
                    # Texto
                    cv2.putText(debug_img, label, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            except Exception as e:
                rospy.logwarn(f"Error processing detection: {e}")

        # ===========================
        # 4. PUBLISH 3D RESULTS
        # ===========================
        self.pub_point.publish(out_msg)

        if debug_img is not None:
            try:
                # # 1. PRIMERO: Sacamos las medidas de la imagen de OpenCV (debug_img)
                # alto, ancho = debug_img.shape[:2]

                # # 2. Pintamos el círculo ROJO en la imagen de OpenCV
                # cv2.circle(debug_img, (ancho // 2, alto // 2), 10, (0, 0, 255), -1)

                # # 3. Chivato por consola
                # rospy.loginfo(f"CHIVATO RESOLUCIÓN: Ancho={ancho}, Alto={alto}")

                # 4. ÚLTIMO PASO: Empaquetamos en formato ROS y publicamos
                img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
                self.pub_debug_img.publish(img_msg)

            except Exception as e:
                rospy.logwarn(f"Error publicando imagen: {e}")

        # ===========================
        # 5. GENERATE & PUBLISH FUSED MASK
        # ===========================
        try:
            # Nota: 'scale' y 'mask_all' ya los tenemos del image_callback, 
            # aquí solo nos importa 'mask_yolo' (la fusionada).
            _, _, _, mask_yolo, _ = self.estimator.get_scale_and_images(
                self.current_image, 
                real_area_m2=(0.015*0.015), 
                yolo_bboxes=boxes_pixels_for_estimator # <--- Le pasamos la lista que llenamos arriba
            )

            if mask_yolo is not None:
                # Publicamos la máscara binaria limpia del agujero detectado
                self.mask_yolo_pub.publish(self.bridge.cv2_to_imgmsg(mask_yolo, encoding="mono8"))
                
        except Exception as e:
            rospy.logwarn(f"Error in Fusion Mask Generation: {e}")

        
        # Useful Info
        rospy.loginfo_throttle(2, f"Publicadas {len(out_msg.detections)} detecciones 3D (Max config: {MAX_DETECTIONS}). Z ref: {self.latest_z:.2f}m")


if __name__ == '__main__':
    try:
        node = NetHoleDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass