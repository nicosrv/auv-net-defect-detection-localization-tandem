#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from net_hole_detector.srv import Trigger, TriggerResponse

class ClaheEnhancer:
    def __init__(self):
        rospy.init_node('clahe_enhancer')
        
        # Parametros
        self.clip_limit = rospy.get_param("~clip_limit", 2.0)
        self.grid_size = int(rospy.get_param("~grid_size", 8))
        
        # Variables initialization
        self.__is_inference_enabled = True
        self.__is_blurring_enabled = False

        # Objetos
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.grid_size, self.grid_size))
        
        # Suscriptor (Topic de entrada se define en el launch)
        self.sub = rospy.Subscriber("image_in", Image, self.callback)
        
        # Publicador (Topic de salida se define en el launch)
        self.pub = rospy.Publisher("image_out", Image, queue_size=1)
        
        # Initialize services
        activation_service = rospy.Service('net_hole_detector/inference_activation_srv', Trigger, self.activate_inference)
        deactivation_service = rospy.Service('net_hole_detector/inference_deactivation_srv', Trigger, self.deactivate_inference)

        activate_blurring_service = rospy.Service('net_hole_detector/blurring_activation_srv', Trigger, self.activate_blurring)
        deactivate_blurring_service = rospy.Service('net_hole_detector/blurring_deactivation_srv', Trigger, self.deactivate_blurring)

        rospy.loginfo(f"CLAHE Node Started. Clip: {self.clip_limit}, Grid: {self.grid_size}")


    """

    """
    def activate_blurring(self, req):
        rospy.loginfo('Blurring activated!')
        self.__is_blurring_enabled = True
        response = TriggerResponse()
        response.success = True
        response.message = "Blurring activated"
        return response
    
    """

    """
    def deactivate_blurring(self, req):
        rospy.loginfo('Blurring deactivated!')
        self.__is_blurring_enabled = False
        response = TriggerResponse()
        response.success = True
        response.message = "Blurring deactivated"
        return response

    """

    """
    def activate_inference(self, req):
        rospy.loginfo('Inference activated!')
        self.__is_inference_enabled = True
        response = TriggerResponse()
        response.success = True
        response.message = "Inference activated"
        return response
    
    """

    """
    def deactivate_inference(self, req):
        rospy.loginfo('Inference deactivated!')
        self.__is_inference_enabled = False
        response = TriggerResponse()
        response.success = True
        response.message = "Inference deactivated"
        return response

    def callback(self, msg):
        try:
            if self.__is_inference_enabled:
                # 1. ROS -> OpenCV
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
                # 2. Aplicar CLAHE (Lab channel L)
                lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = self.clahe.apply(l)
                lab = cv2.merge((l2, a, b))
                enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                if self.__is_blurring_enabled:
                    enhanced_img = cv2.blur(enhanced_img, (7, 7))

                # 3. OpenCV -> ROS
                out_msg = self.bridge.cv2_to_imgmsg(enhanced_img, encoding="bgr8")
                out_msg.header = msg.header # Importante mantener el timestamp original
                
                self.pub.publish(out_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CLAHE Error: {e}")

if __name__ == '__main__':
    ClaheEnhancer()
    rospy.spin()