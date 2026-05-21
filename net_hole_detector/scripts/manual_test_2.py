# import cv2
# import numpy as np
# import os
# from net_hole_detector.camera_geometry import CameraGeometry
# from net_hole_detector.scale_estimator import ScaleEstimator

# # --- CONFIGURACIÓN ---
# # Ajusta estas rutas a tus archivos reales
# YAML_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/right_laser.yaml"
# IMAGE_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/frame_002366.jpg" # Usa una de las que ya tienes
# BBOX_TXT_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/frame_002366.txt"   # Crea este archivo con una línea YOLO

# def main():
#     # 1. Iniciar Geometría con el YAML
#     print("--- 1. Cargando Geometría ---")
#     geo = CameraGeometry(yaml_path=YAML_PATH)
    
#     # ----- Forcing use of P Matrix -----
#     print("--- 1.5 Seleccionando Matriz de Proyección (Rectified Mode) ---")
#     geo.select_matrix(is_rectified_topic=True)
#     # ------------------------

#     # 2. Cargar Imagen
#     print(f"--- 2. Cargando Imagen: {IMAGE_PATH} ---")
#     img = cv2.imread(IMAGE_PATH)
#     if img is None:
#         print("Error cargando imagen")
#         return

#     # Actualizamos las dimensiones en la clase por seguridad (por si el YAML miente)
#     h, w = img.shape[:2]
#     geo.img_h = h
#     geo.img_w = w

#     # NOTA: NO hacemos undistort, porque la imagen ya es rectificada.
#     # img ya está lista para usarse.

#     # # 3. CORREGIR DISTORSIÓN (Importante para que las líneas sean rectas)
#     # # Fíjate cómo cambia la imagen en los bordes
#     # img_undistorted = geo.undistort_image(img)

#     # 4. Iniciar Estimador de Escala
#     print("--- 3. Calculando Escala (Z) ---")
#     estimator = ScaleEstimator()
#     # Asumimos area real basada en lado 1.5cm
#     real_area = 0.015 * 0.015 
#     scale = estimator.get_scale_from_blobs(img, real_area_m2=real_area, debug=False)

#     if scale is None:
#         print("No se detectaron blobs para calcular escala.")
#         return

#     z_depth = geo.get_z_distance(scale)
#     print(f"Escala: {scale:.5f} m/px")
#     print(f"DISTANCIA Z DETECTADA: {z_depth:.3f} metros")

#     # 5. Leer Bounding Box (Simulando YOLO)
#     print("--- 4. Proyectando Bounding Box ---")
#     if os.path.exists(BBOX_TXT_PATH):
#         with open(BBOX_TXT_PATH, 'r') as f:
#             line = f.readline().strip()
#             # Parsear: class x_n y_n w_n h_n
#             parts = list(map(float, line.split()))
#             yolo_bbox = parts  # [clase, x, y, w, h]

#             # A) Obtener coordenadas pixel del CENTRO
#             u, v = geo.yolo_to_pixels(yolo_bbox)
            
#             # B) Obtener coordenadas 3D (X, Y, Z)
#             # Usamos la Z calculada antes
#             point_3d = geo.project_pixel_to_3d(u, v, z_depth)
            
#             print(f"BBox Centro (px): ({u:.1f}, {v:.1f})")
#             print(f"POSICIÓN REAL 3D: X={point_3d[0]:.3f}m, Y={point_3d[1]:.3f}m, Z={point_3d[2]:.3f}m")

#             # C) DIBUJAR EN LA IMAGEN
#             # Convertimos a esquinas para dibujar el recuadro
#             x1, y1, x2, y2 = geo.get_bbox_corners_pixels(yolo_bbox)
            
#             # Dibujamos en la imagen corregida
#             # Rectángulo Azul: La BBox
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             # Punto Rojo: El Centro calculado
#             cv2.circle(img, (int(u), int(v)), 5, (0, 0, 255), -1)
            
#             # Texto con coordenadas
#             text = f"X:{point_3d[0]:.2f} Y:{point_3d[1]:.2f} Z:{point_3d[2]:.2f}"
#             cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             cv2.imshow("Integration Test (Rectified)", img)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#     else:
#         print(f"No se encontró {BBOX_TXT_PATH}. Crea el archivo con datos YOLO para probar.")

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import os
from net_hole_detector.camera_geometry import CameraGeometry
from net_hole_detector.scale_estimator import ScaleEstimator

# --- CONFIGURACIÓN ---
YAML_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/right_laser.yaml"
IMAGE_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/frame_002366.jpg"
BBOX_TXT_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/frame_002366.txt"

def main():
    # 1. Cargar todo
    geo = CameraGeometry(yaml_path=YAML_PATH)
    geo.select_matrix(is_rectified_topic=True) # IMPORTANTE: Modo Rectificado

    img = cv2.imread(IMAGE_PATH)
    if img is None: return

    h, w = img.shape[:2]
    geo.img_h = h
    geo.img_w = w

    # 2. Calcular Escala y Z (Dummy o Real)
    estimator = ScaleEstimator()
    real_area = 0.015 * 0.015 
    scale = estimator.get_scale_from_blobs(img, real_area_m2=real_area, debug=False)
    
    if scale:
        z_depth = geo.get_z_distance(scale)
    else:
        z_depth = 0.63 # Usamos el valor que te dio antes para no fallar
        print("[WARN] Usando Z fija para visualización")

    # 3. VISUALIZACIÓN DE CENTROS
    print("\n" + "="*50)
    print("COMPARACIÓN DE CENTROS")
    print("="*50)
    
    # A) Centro Geométrico (Mitad de la imagen)
    geo_cx = int(w / 2)
    geo_cy = int(h / 2)
    print(f"1. Centro IMAGEN (Blanco): ({geo_cx}, {geo_cy})")

    # B) Centro Óptico (De la calibración P)
    opt_cx = int(geo.cx)
    opt_cy = int(geo.cy)
    print(f"2. Centro LENTE  (Verde):  ({opt_cx}, {opt_cy})")
    
    diff_x = opt_cx - geo_cx
    diff_y = opt_cy - geo_cy
    print(f"--> DESVIACIÓN: La lente está desplazada {diff_x}px en X y {diff_y}px en Y respecto al centro.")
    print("="*50 + "\n")

    # --- DIBUJADO ---
    
    # 1. Dibujar Centro Geométrico (Cruz Blanca Grande)
    # Eje Horizontal
    cv2.line(img, (0, geo_cy), (w, geo_cy), (255, 255, 255), 1) 
    # Eje Vertical
    cv2.line(img, (geo_cx, 0), (geo_cx, h), (255, 255, 255), 1)
    cv2.putText(img, "Centro Imagen", (50, geo_cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 2. Dibujar Centro Óptico (Círculo Verde Relleno)
    cv2.circle(img, (opt_cx, opt_cy), 8, (0, 255, 0), -1) 
    cv2.putText(img, "Centro Lente (CX,CY)", (opt_cx + 15, opt_cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3. Dibujar Objeto Detectado (Si hay BBox)
    if os.path.exists(BBOX_TXT_PATH):
        with open(BBOX_TXT_PATH, 'r') as f:
            line = f.readline().strip()
            parts = list(map(float, line.split()))
            yolo_bbox = parts

            u, v = geo.yolo_to_pixels(yolo_bbox)
            
            # Punto Rojo del objeto
            cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)
            
            # Línea desde el Centro Óptico hasta el Objeto (El vector real)
            cv2.line(img, (opt_cx, opt_cy), (int(u), int(v)), (0, 255, 255), 2) # Línea amarilla
            
            # Texto coordenadas 3D
            point_3d = geo.project_pixel_to_3d(u, v, z_depth)
            
            # Caja alrededor
            x1, y1, x2, y2 = geo.get_bbox_corners_pixels(yolo_bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            text_pos = f"X:{point_3d[0]:.2f} Y:{point_3d[1]:.2f}"
            cv2.putText(img, text_pos, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Mostrar
    cv2.imshow("Centros: Real vs Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






