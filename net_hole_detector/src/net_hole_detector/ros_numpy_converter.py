import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image

class RosNumPyConverter:
    """
    Clase que sustituye a CvBridge para evitar errores de librerías dinámicas (libffi)
    en entornos Conda. Usa puramente NumPy.
    """
    def __init__(self):
        pass

    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):
        """
        Convierte sensor_msgs/Image a OpenCV (NumPy array).
        """
        dtype = np.uint8
        n_channels = 1

        # Deducir canales según el encoding del mensaje
        if '8' in img_msg.encoding:
            dtype = np.uint8
        elif '16' in img_msg.encoding:
            dtype = np.uint16
            
        if 'rgb' in img_msg.encoding or 'bgr' in img_msg.encoding:
            n_channels = 3
        
        # Convertir bytes a array
        # buffer es el array plano de bytes
        im_arr = np.frombuffer(img_msg.data, dtype=dtype)
        
        # Darle forma (Alto, Ancho, Canales)
        if n_channels == 3:
            im_arr = im_arr.reshape((img_msg.height, img_msg.width, n_channels))
        else:
            im_arr = im_arr.reshape((img_msg.height, img_msg.width))

        # Conversión de color si se pide explícitamente y es necesario
        # (Nota: Asumimos que si llega bgr8 y quieres bgr8, no hacemos nada)
        if desired_encoding == "bgr8" and "rgb" in img_msg.encoding:
            return cv2.cvtColor(im_arr, cv2.COLOR_RGB2BGR)
        
        if desired_encoding == "rgb8" and "bgr" in img_msg.encoding:
            return cv2.cvtColor(im_arr, cv2.COLOR_BGR2RGB)

        return im_arr

    def cv2_to_imgmsg(self, cv_img, encoding="passthrough"):
        """
        Convierte OpenCV (NumPy array) a sensor_msgs/Image.
        """
        img_msg = Image()
        img_msg.height = cv_img.shape[0]
        img_msg.width = cv_img.shape[1]
        
        if len(cv_img.shape) == 3:
            # Es color (B G R)
            img_msg.encoding = "bgr8" if encoding == "passthrough" else encoding
            img_msg.step = cv_img.shape[1] * 3  # Ancho * bytes por pixel
        else:
            # Es escala de grises
            img_msg.encoding = "mono8" if encoding == "passthrough" else encoding
            img_msg.step = cv_img.shape[1]      # Ancho * 1 byte

        img_msg.data = cv_img.tobytes()
        img_msg.header.stamp =  rospy.Time.now() # Ojo: mejor copiar el header original si se puede
        img_msg.header.frame_id = "camera_frame" # Placeholder
        
        return img_msg