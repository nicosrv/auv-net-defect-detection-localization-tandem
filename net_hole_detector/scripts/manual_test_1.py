#!/usr/bin/env python3
import cv2
import numpy as np
from net_hole_detector.scale_estimator import ScaleEstimator

# NOMBRE DE TU IMAGEN
# IMAGE_PATH = "/home/xiroinuc/Escritorio/pruebas_tandem/pruebas_scale_estimator/frame_002366.jpg"
# IMAGE_PATH = "/home/xiroinuc/Escritorio/bagfiles_uji/images_uji/gripper_camera_net/frame_001161.jpg"
# IMAGE_PATH = "/home/xiroinuc/Escritorio/bagfiles_uji/images_uji/gripper_camera_net/frame_000737.jpg"
IMAGE_PATH = "/home/xiroinuc/Escritorio/bagfiles_uji/images_uji/gripper_camera_net/frame_003348.jpg"
# IMAGE_PATH = "/home/xiroinuc/Escritorio/bagfiles_uji/images_uji/gripper_camera_net/frame_001865.jpg"



def main():
    # 1. Cargar imagen
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: No encuentro la imagen. Revisa el nombre.")
        return

    print(f"Imagen cargada: {img.shape}")

    # 2. Inicializar estimador
    estimator = ScaleEstimator()

    # 3. Probar detección de red (HEXÁGONOS)
    print("\n--- Ejecutando Detección de Blobs (Red) ---")
    
    # Activamos debug=True para que genere las fotos
    scale = estimator.get_scale_from_blobs(img, real_area_m2=0.000225, debug=True)

    if scale:
        print("\nRESULTADOS:")
        print(f"Escala calculada: {scale:.6f} metros/pixel")
        
        # Simulación de cálculo de Z
        # Supongamos FOCAL = 1000 píxeles (dato inventado hasta tener calibración)
        focal_dummy = 1000 
        Z = focal_dummy * scale
        
        print(f"Distancia Z estimada (con focal 1000): {Z:.3f} metros")
        print("Revisa las imágenes 'debug_*.jpg' que se han creado.")
    else:
        print("Fallo: No se detectó patrón de red válido.")

if __name__ == "__main__":
    main()