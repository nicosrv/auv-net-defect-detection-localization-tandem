#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from ultralytics import YOLO

# Import new messages.
from net_hole_detector.msg import BoundingBox, BoundingBoxArray

# Import auxiliary classes.
from net_hole_detector.scale_estimator import ScaleEstimator
from net_hole_detector.ros_numpy_converter import RosNumPyConverter

class BboxDetector:
    def __init__(self):
        rospy.init_node('bbox_detector')

        # --- 1. PARAMETERS ---
        self.model_path = rospy.get_param("~pathWeights")
        self.conf_thres = rospy.get_param("~confidenceThreshold", 0.5)
        self.period = float(rospy.get_param("~period", 0.5))
        # Variable to store last time we process an image
        self.last_process_time = rospy.Time(0)
        # self.input_topic = rospy.get_param("~input_topic", "/image_rect_color")

        # --- 2. UPLOAD MODEL ---
        rospy.loginfo(f"Loading YOLO from: {self.model_path} ...")
        self.model = YOLO(self.model_path)
        rospy.loginfo("Model loaded and ready.")

        # --- 3. UPLOAD CLASSES ---
        self.scale_estimator = ScaleEstimator()
        self.bridge = RosNumPyConverter()

        # --- 4. SUBSCRIBER AND PUBLISHER ---
        self.sub_img = rospy.Subscriber('camera_input', Image, self.callback_image, queue_size=1)
        self.pub_det = rospy.Publisher('yolo/detections', BoundingBoxArray, queue_size=1)

        # --- 5. DEBUG IMAGE PUBLISHERS ---
        self.pub_blob_bin = rospy.Publisher('/net_hole_detector/blobs/binary', Image, queue_size=1)
        self.pub_blob_overlay = rospy.Publisher('/net_hole_detector/blobs/overlay', Image, queue_size=1)

    def callback_image(self, msg):
        # THROTTLE
        now = rospy.Time.now()

        # Calculate time since last processed image.
        if (now - self.last_process_time).to_sec() < self.period:
            # if less time passed we ignore the image.
            return
        
        # else, we update the clock.
        self.last_process_time = now
        
        try:
            # 1. Imagen ROS -> Numpy (Using auxiliary class bridge).
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 2. Process BLOBS.
            _, _, mask_img, _, overlay_img = self.scale_estimator.get_scale_and_images(cv_img)

            # If a net has been detected, we publish the debug images.
            if mask_img is not None:
               # Publish binary (mono8 because it is gray).
                self.pub_blob_bin.publish(self.bridge.cv2_to_imgmsg(mask_img, "mono8"))
                # Publish overlay (bgr8 because it has colors).
                self.pub_blob_overlay.publish(self.bridge.cv2_to_imgmsg(overlay_img, "bgr8"))

            # 3. Inference
                # verbose=False, so it does not fill the console with text.
            results = self.model(cv_img, verbose=False, conf=self.conf_thres)
            
           # 4. Prepare Output Message.
            msg_out = BoundingBoxArray()
            msg_out.header = msg.header # WE COPY THE ORIGINAL TIMESTAMP
            msg_out.boxes = []
            
           # 5. Fill data
            result = results[0]

            if len(result.boxes) > 0:
                # xywhn returns: x_center, y_center, width, height (NORMALIZED 0-1).
                boxes_data = result.boxes.xywhn.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes_data)):
                    bbox = BoundingBox()
                    bbox.class_id = str(int(classes[i]))
                    bbox.score = float(scores[i])
                    
                    # Normalized coordinates (0 - 1)
                    bbox.x = float(boxes_data[i][0]) # Center X
                    bbox.y = float(boxes_data[i][1]) # Center Y
                    bbox.w = float(boxes_data[i][2]) # Width
                    bbox.h = float(boxes_data[i][3]) # Heigh
                    
                    msg_out.boxes.append(bbox)
            
            # 6. Publish.
            self.pub_det.publish(msg_out)

        except Exception as e:
            rospy.logerr(f"Error YOLO: {e}")

if __name__ == '__main__':
    try:
        BboxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
