#!/usr/bin/env python3
import cv2
import numpy as np
import rospy


class ScaleEstimator:

    def __init__(self):
        # VISION PARAMETERS
        raw_block_sz = rospy.get_param('~thresh_block_size', 41)
        self.thresh_c = rospy.get_param('~thresh_c', 5)

        # NOISE FILTERS
        raw_blur_k    = rospy.get_param('~blur_kernel', 7)
        raw_morph_k   = rospy.get_param('~morph_kernel', 5)

        # GEOMETRIC FILTERS
        self.min_area          = rospy.get_param('~min_area_px', 25)
        self.max_percent_area  = rospy.get_param('~max_percent_area', 0.1)

        # =========================================
        # SANITY CHECK (Force ODD NUMBERS)
        # =========================================
        self.thresh_block_size = raw_block_sz + 1 if raw_block_sz % 2 == 0 else raw_block_sz
        self.blur_k            = raw_blur_k + 1   if raw_blur_k % 2 == 0 else raw_blur_k
        self.morph_k           = raw_morph_k + 1  if raw_morph_k % 2 == 0 else raw_morph_k

        # Logging with info.
        rospy.loginfo(f"[ScaleEstimator] Config Loaded: Blur={self.blur_k}, ThreshBlock={self.thresh_block_size}, C={self.thresh_c}")


    """
    Function: get_scale_and_images

    Calculates Meters/Pixel scale based in SQUARE ROOT of AREA of blobs
    This is more robust than using the width. With a black net and blue background
    Also generates TWO masks: one with all valid blobs, one with filtered blobs (with bboxes)
    and also an overlay image.

    Parameters
    ----------
        real_area_m2: float
            Real area of the Net in square meters
            If square size = 1.5cm -> 0.015 * 0.015 = 0.000225 m2
        yolo_bboxes = List[Tuple]
            List of tuples with format (x,y,w,h)

    Returns
    --------
        scale
            meters/pixel ratio
        median_area_px
            median are of holes/blobs in pixels
        mask_all
            mask with all blobs detected
        mask_yolo
            mask with blobs inside bbox detected by yolo
        overlay_img
            original image with blobs detected drawn over it
    """
    def get_scale_and_images(self, cv_image, real_area_m2=0.000225, yolo_bboxes=[]):
        
        # 1. Split channels
        # If the background is blue, the Blue channel will have the maximum contrast
        # between the net (dark) and the background (light).
        if len(cv_image.shape) == 3:
            blue = cv_image[:, :, 0]
        else:
            blue = cv_image
        blurred = cv2.GaussianBlur(blue, (self.blur_k, self.blur_k), 0)

        # 2. Thresholding
        # We want the image to be either black or white, to do so 
        # everything 'blackish' will be black and everything 'whiteish' will be white.
        #_, thresh = cv2.threshold(blue, 50, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, self.thresh_block_size, self.thresh_c)

        # Optional: Noise removal
        kernel = np.ones((self.morph_k,self.morph_k), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 3. Find contours
        contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sqrt_areas = []
        valid_contours = [] # Just to publish them
        yolo_match_contours = [] # Only the contours matching with YOLO bbox

        # Find image dimensions, to reject huge object
        img_h, img_w = blue.shape

        # Safety check -> yolo_bboxes always a list
        if yolo_bboxes is None:
            yolo_bboxes = []

        for cnt in contours:
            # First we calculate the area
            area_px = cv2.contourArea(cnt)
            
            # --- FILTERS ---
            
            # 3.1. Size Filter
            # Reject small noise (parametrized)
            # Reject huge breakages (> 10% image width)
            if area_px < self.min_area or area_px > (img_w * img_h * self.max_percent_area):
                continue

            # 3.2. Aspect Ratio Filter (Form):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # IF IS VALID:
            # We use the square root of the area as our lineal metric "L"
            # L_px = sqrt(Area_px)
            l_px = np.sqrt(area_px)
            sqrt_areas.append(l_px)

            valid_contours.append(cnt)

            # --- MULTIPLE BBOXES ---
            # Check if this contour lays in ANY of the YOLO bboxes
            if yolo_bboxes:
                # Calculate centroid of the blob
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Check ALL BBOXES
                    for box in yolo_bboxes:
                        bx, by, bw, bh = box
                        # Safety margin, if the center is inside the box, we append it
                        if (bx < cX < bx + bw) and (by < cY < by + bh):
                            yolo_match_contours.append(cnt)
                            break



        if not sqrt_areas:
            return None, None, None, None, None
        
        # 4. Using MEDIAN to avoid Damaged Holes (that passed the filter) to affect the calculations
        # we call this the "Characteristic Longitude"
        median_l_px = np.median(sqrt_areas)

        # Calculate the real "Characteristic Longitude" (size of the equivalente square)
        real_l_meters = np.sqrt(real_area_m2)

        # 5. Image generation (in memory)

        # Draw over original image detected contoursin RED
        overlay_img = cv_image.copy()
        cv2.drawContours(overlay_img, valid_contours, -1, (0, 255, 0), 3)

        # Mask 1: ALL BLOBS (for debug) White backgorund, Black holes
        mask_all = np.ones_like(blue, dtype=np.uint8)
        cv2.drawContours(mask_all, valid_contours, -1, 255, -1)

        # Mask 2: ONLY in BBOXES
        mask_yolo = np.zeros_like(blue, dtype=np.uint8)
        if yolo_match_contours:
            cv2.drawContours(mask_yolo, yolo_match_contours, -1, 255, -1)

        #print(f"[DEBUG] Valid hexagons: {len(sqrt_areas)}. Median: {median_l_px:.2f} px")
        
        # Unified return (Meters / Pixel)
        # scale = L_real / L_pixel
        scale = real_l_meters / median_l_px
        median_area_px = median_l_px ** 2

        return scale, median_area_px, mask_all, mask_yolo, overlay_img


    """
    Function: get_scale_from_laser_lines

    Detects two parallel laser lines and calculates the scale.
    Assumes the lasers are much brighter than the rest.

    Parameters
    ----------
        cv_image
            Image to get the scale from
        real_dist_meters
            Real distance between the lines

    Returns
    ---------
        float
            meters/pixel ratio between the lasers
    """
    def get_scale_from_laser_lines(self, cv_image, real_dist_meters) -> float:

        # 1. Color Mask --> ADJUST TO LASER COLOR (Assuming Red Laser, adjust if it is Green)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red Ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Clean the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 2. Detect Lines with Hough
        # rho=1 pixel, theta=1 degree, threshold=minimum intersections
        lines = cv2.HoughLines(mask, 1, np.pi / 180, 100)
        
        if lines is None:
            return None
        
        # lines comes with the form [[rho, theta], [rho, theta], ...]
        # we want to group them by angle. to find the parallel ones

        valid_rhos = []
        target_theta = None

        # Step A: Find dominant angle
        # (We Assume lasers are the strongest/longest lines)
        thetas = lines[:, 0, 1]

        # Angle histogram to find the dominant one, convert to degrees
        degrees = np.rad2deg(thetas)
        hist, bin_edges = np.histogram(degrees, bins=range(0, 180, 5))
        peak_bin = np.argmax(hist)
        dominant_angle_min = bin_edges[peak_bin]
        dominant_angle_max = bin_edges[peak_bin+1]

        # Step B: Stay only with the lines with this angle
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            
            if dominant_angle_min <= angle_deg <= dominant_angle_max:
                valid_rhos.append(rho)
        
        if len(valid_rhos) < 2:
            return None
        
        # STEP C: Search for the two distance groups (Rho)
        # We use simple K-Means or order and find the gap
        valid_rhos.sort()
        
        # Search for the biggest gap between the detected lines
        # This split left and right lasers
        max_gap = 0
        split_idx = 0

        for i in range(len(valid_rhos) - 1):
            gap = valid_rhos[i+1] - valid_rhos[i]
            if gap > max_gap:
                max_gap = gap
                split_idx = i

        # If the gap is too small (10px), we only detected ONE laser (with double line)
        if max_gap < 20:
            return None
        
        # Average rho from group 1 and 2
        rho_group1 = np.mean(valid_rhos[:split_idx+1])
        rho_group2 = np.mean(valid_rhos[split_idx+1:])
        
        dist_pixels = abs(rho_group1 - rho_group2)
        
        return real_dist_meters / dist_pixels

