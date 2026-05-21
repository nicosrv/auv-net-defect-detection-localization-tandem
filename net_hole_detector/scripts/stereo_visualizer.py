#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from net_hole_detector.msg import BoundingBoxArray, Detection3DArray

class StereoVisualizer:
    def __init__(self):
        rospy.init_node('stereo_visualizer')
        self.bridge = CvBridge()

        # Suscriptores
        sub_img = Subscriber("/girona500/xiroi/stereo_ch3/left_optical/image_color", Image)
        sub_bb  = Subscriber("/net_hole_detector/stereo_left/bounding_boxes", BoundingBoxArray)
        # Sincronizamos para que la caja que pintamos sea la que corresponde a ese frame exacto
        self.ats = ApproximateTimeSynchronizer([sub_img, sub_bb], queue_size=10, slop=0.1)
        self.ats.registerCallback(self.callback)

        # Publisher de la imagen pintada
        self.pub_debug = rospy.Publisher("/net_hole_detector/debug_image", Image, queue_size=1)

    def callback(self, img_msg, bb_msg):
        try:
            # Convertir a OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            h, w, _ = cv_img.shape

            # === DATOS HARDCODED DE TU CAMERA_INFO PARA EL DEBUG ===
            cx, cy = 640.0, 360.0
            fx, fy = 736.235, 736.235
            
            # === DIBUJAR ESTRUCTURA DE DEBUG ===
            # 1. Dibujar Centro Óptico (Cruz Roja)
            cv2.drawMarker(cv_img, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(cv_img, "Optical Center (cx,cy)", (int(cx)+10, int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for bb in bb_msg.boxes:
                # 2. Calcular Centro de la Caja (Verde)
                u = int((bb.x + bb.w / 2.0) * w) # Asegúrate de usar .w/.width según tu .msg
                v = int((bb.y + bb.h / 2.0) * h)

                # 3. Dibujar Punto del Centro de Detección (Verde)
                cv2.circle(cv_img, (u, v), 5, (0, 255, 0), -1)

                # 4. Dibujar Línea de Vector de Error (Amarilla)
                cv2.line(cv_img, (int(cx), int(cy)), (u, v), (0, 255, 255), 1)

                # 5. Dibujar Rectángulo de la Caja (Gris flojo)
                x1 = int(bb.x * w)
                y1 = int(bb.y * h)
                bw = int(bb.w * w)
                bh = int(bb.h * h)
                cv2.rectangle(cv_img, (x1, y1), (x1 + bw, y1 + bh), (150, 150, 150), 1)
                
                # === CÁLCULO DE DEBUG EN PÍXELES ===
                # Esto te dirá el signo y magnitud reales
                pixel_dx = u - cx
                pixel_dy = v - cy
                
                # Etiqueta de Debug en Píxeles
                debug_label = f"dx:{pixel_dx:.0f}px, dy:{pixel_dy:.0f}px"
                cv2.putText(cv_img, debug_label, (u + 10, v + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Publicar resultado
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        except Exception as e:
            rospy.logerr(f"Visualizer Error: {e}")

if __name__ == '__main__':
    StereoVisualizer()
    rospy.spin()