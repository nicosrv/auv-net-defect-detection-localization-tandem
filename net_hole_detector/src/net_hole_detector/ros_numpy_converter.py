import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image

# Module used to convert ROS Image messages to NumPy/OpenCV images
# without using CvBridge. This is useful for the YOLO node because it runs inside
# a Conda environment, where CvBridge can cause dynamic library conflicts. The
# converter keeps the image data in a format that can be processed by OpenCV and YOLO.

class RosNumPyConverter:
    
    #Replacement class for CvBridge to prevent dynamic library errors (libffi)
    #within Conda environments. Built purely using NumPy.
    
    def __init__(self):
        pass

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        
        """
        Converts sensor_msgs/Image to OpenCV (NumPy array).
        """

        dtype = np.uint8
        n_channels = 1

        # Deduce channels based on the message encoding.
        if '8' in img_msg.encoding:
            dtype = np.uint8
        elif '16' in img_msg.encoding:
            dtype = np.uint16
            
        if 'rgb' in img_msg.encoding or 'bgr' in img_msg.encoding:
            n_channels = 3
        
        # Convert bytes to array
        # buffer is the flat array of bytes.
        im_arr = np.frombuffer(img_msg.data, dtype=dtype)
        
        # Reshape (Height, Width, Channels).
        if n_channels == 3:
            im_arr = im_arr.reshape((img_msg.height, img_msg.width, n_channels))
        else:
            im_arr = im_arr.reshape((img_msg.height, img_msg.width))

       # Perform color conversion only when explicitly required.
        if desired_encoding == "bgr8" and "rgb" in img_msg.encoding:
            return cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
        
        if desired_encoding == "rgb8" and "bgr" in img_msg.encoding:
            return cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)

        return im_arr

    