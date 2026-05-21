#!/usr/bin/env python3
import numpy as np
import cv2
import yaml
import os

class CameraGeometry:
    r"""
    Class to manage 2D -> 3D projection
    STRATEGY:
    1. Charges default values of Calibration Matrix from a .yaml file (Plan B)
    2. If a CameraInfo message arrives, actualizes this values (Plan A)

    Parameters
    ----------
        yaml_path
            path to .yaml file to load the camera configuration
    """
    def __init__(self, yaml_path=None):
        # Raw matrix containers
        self.k_matrix = None      # Intrinsec Matrix (3x3) -> For RAW
        self.p_matrix = None      # Projection Matrix (3x3 recortada) -> For RECT
        self.dist_coeffs = None   # Distorsion Coefficients
        
        # "Active" Matrix used for calculus (fx, fy...)
        self.camera_matrix = None 

        # Actual Values
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.img_w = None
        self.img_h = None

        # Calibration received flag
        self.is_calibrated = False

        # Flag to only print once
        self.has_printed_info = False
        
        # Try to load configuration immediately
        if yaml_path:
            self.load_from_yaml_file(yaml_path)

        # If we failed to parse yaml file, we WARN
        if not self.is_calibrated:
            print("[CameraGeometry] WARNING: Initialized without calibration. Waiting for YAML or ROS Topic.")


    """
    Function: load_from_yaml_file

    """
    def load_from_yaml_file(self, file_path):
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


    """
    Function: set_camera_info

    """
    def set_camera_info(self, camera_info_msg):
        """
        Overwrittes configuration if a live ROS message arrives
        Store K (for Raw) and P (for Rect)
        """
        # Extract intrinsec parameters
        # Stores distorsion coefficients
        self.dist_coeffs = np.array(camera_info_msg.D)

        # 1. Store K MATRIX (3x3) -> For RAW
        self.k_matrix = np.array(camera_info_msg.K).reshape(3, 3)

        # 2. Store P MATRIX (3x3) -> For Rect
        p_full = np.array(camera_info_msg.P).reshape(3, 4)
        self.p_matrix = p_full[:3, :3] # stay with 3x3 left side

        self.is_calibrated = True

        self.img_w = camera_info_msg.width
        self.img_h = camera_info_msg.height

        if not self.has_printed_info:
            # Using K Matrix to print directly, in case self.fx is None if YAML was not loaded
            fx_temp = self.k_matrix[0, 0]
            fy_temp = self.k_matrix[1, 1]
            print(f"[CameraGeometry] ¡RECEIVED LIVE CALIBRATION! FX(Raw): {fx_temp:.1f}, FY(Raw): {fy_temp:.1f}")
            self.has_printed_info = True


    """
    Function: select_matrix

    """
    def select_matrix(self, is_rectified_topic):
        """
        Sets fx, fy, cx, cy according to image type
        Rect -> P, Raw -> K
        """
        if not self.is_calibrated:
            return

        if is_rectified_topic:
            # Option A: Rectified Image -> Using P
            matrix = self.p_matrix
        else:
            # Option B: Raw Image -> Using K
            matrix = self.k_matrix

        # If P is None (Incomplete YAML), fallback to K
        if matrix is None: 
            matrix = self.k_matrix

        # Update values
        self.fx = matrix[0, 0]
        self.fy = matrix[1, 1]
        self.cx = matrix[0, 2]
        self.cy = matrix[1, 2]
        
        # Stores Active Matrix
        self.camera_matrix = matrix


    """
    Function: yolo_to_pixels
    
    """
    def yolo_to_pixels(self, yolo_bbox) -> tuple:
        r"""
        Step 1: Transforms YOLO normalized format to image pixels

        Parameters
        ----------
            yolo_bbox
                [class_id, x_center_norm, y_center_norm, width_norm, height_norm]

        Returns
        ---------
            tuple
                (u, v) -> Image center in pixels
        """
        # SECURITY: If we are not calibrated, we assume 1280x720 to avoid crashing
        w_total = self.img_w if self.img_w else 1280
        h_total = self.img_h if self.img_h else 720

        # Unpacking
        _, x_n, y_n, _, _ = yolo_bbox

        # Multiplying percentage with image total size
        u = x_n * w_total       # X center in pixels
        v = y_n * h_total       # Y center in pixels

        return u, v
    

    """
    Function: get_z_distance
    
    """
    def get_z_distance(self, scale_meters_per_pixel):
        """
        Step 2: Calculate Depth (Z) using NET/SCALE 

        """
        if not self.is_calibrated or scale_meters_per_pixel is None or self.fx is None:
            return None

        # Z = focal * (Meters / Pixel)
        return self.fx * scale_meters_per_pixel


    """
    Function: project_pixel_to_3d
    
    """
    def project_pixel_to_3d(self, u, v, z_distance):
        """
        Step 3: Having Z, pixel pose (u,v) get X, Y (meters)
        """
        if not self.is_calibrated or z_distance is None:
            # Returning 0s vector to indicate an error
            return np.array([0.0, 0.0, 0.0])
        
        # Inverse Pinhole Formulae
        x_meters = (u - self.cx) * z_distance / self.fx
        y_meters = (v - self.cy) * z_distance / self.fy

        return np.array([x_meters, y_meters, z_distance])


    """
    Function: get_object_dimensions
    
    """
    def get_object_dimensions(self, w_norm, h_norm, z_distance):
        """
        Calculates the real width and hight (in meters) of the detected object
        
        Parameters:
            w_norm
                Normalized width (0.0 to 1.0) from YOLO
            h_norm
                Normalized height (0.0 to 1.0) from YOLO
            z_distance
                Z Distance to the object in meters
        
        Returns:
            (width_m, height_m)
                Tuple with real dimensions
        """
        if not self.is_calibrated or z_distance is None:
            return 0.0, 0.0
        
        # 1. De-Normalization
        img_w = self.img_w if self.img_w else 1280
        img_h = self.img_h if self.img_h else 720

        w_px = w_norm * img_w
        h_px = h_norm * img_h

        # 2. Apply Pinhole Formula: Size_Real = (Size_Pixel * Distancia_Z) / Focal_Length
        width_m = (w_px * z_distance) / self.fx
        height_m = (h_px * z_distance) / self.fy

        return width_m, height_m


    """
    Function: get_bbox_corners_pixels

    """
    def get_bbox_corners_pixels(self, yolo_bbox):
        """
        Returns (x_min, y_min, x_max, y_max) in absolute pixels
        Useful to draw a rectangle with cv2.rectangle
        """
        w_total = self.img_w if self.img_w else 1280
        h_total = self.img_h if self.img_h else 720

        # yolo_bbox = [class, x_c, y_c, w, h] (normalized)
        _, x_c, y_c, w, h = yolo_bbox

        x_min = int((x_c - w / 2) * w_total)
        y_min = int((y_c - h / 2) * h_total)
        x_max = int((x_c + w / 2) * w_total)
        y_max = int((y_c + h / 2) * h_total)

        return x_min, y_min, x_max, y_max
    

    """
    Function: undistort_image
    
    """
    def undistort_image(self, cv_image):
        """
        Corrects Image Distorsion
        ALWAYS uses K and D (Physical parameters), never P.
        """
        if not self.is_calibrated or self.k_matrix is None or self.dist_coeffs is None:
            return cv_image
        
        return cv2.undistort(cv_image, self.k_matrix, self.dist_coeffs)


    
# TESTING BLOCK (Only to execute this file directly)
if __name__ == "__main__":
    print("Executing unitary test from CameraGeometry...")

    # Dummy data to test it does not explode
    K_dummy = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    geom = CameraGeometry(K_dummy, 1920, 1080)

    # Dummy Bounding Box: [class, x, y, w, h]
    bbox_dummy = [0, 0.5, 0.5, 0.1, 0.1]

    px = geom.yolo_to_pixels(bbox_dummy)
    res = geom.calculate_position_3d(px, 0.015) # 1.5 cm

    print(f"Test result: {res}")
    

    # Es esta la matriz que necesito?? No necesito la P con las rotaciones?? Creo que como la cámara es MONO, tampoco hace falta
    # Necesito algún set_image_dimensions?? No hace falta, ya lo hago
    # Podría meterle un publisher a esto, para saber cuando está recibiendo la camera info y las imágenes?? tiene sentido??
    # cuando sino empiezo con el get_z_distance?? Tampoco hace falta