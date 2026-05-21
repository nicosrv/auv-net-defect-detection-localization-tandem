#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera una misión XML de inspección en PLANO VERTICAL frente a una columna,
utilizando maniobras 'SwaySection'.

MODIFICADO:
  - Ahora toma la ubicación de la columna en metros (NED) relativa a un
    origen NED (lat/lon) especificado.
  - Esto evita errores de "distancia demasiado grande" si el origen de la misión
    y el origen NED del robot están en lugares diferentes.

Patrón:
  1) GOTO a esquina superior izquierda (o derecha) del plano
  2) SWAY_SECTION horizontal (L->R o R->L)
  3) SWAY_SECTION vertical (descenso)
  4) Alterna hasta alcanzar la profundidad inferior
"""

import math
import argparse
import xml.etree.ElementTree as ET
from typing import Tuple

EARTH_RADIUS_M = 6371000.0


def meters_to_latlon_offsets(lat0_deg: float, north_m: float, east_m: float) -> Tuple[float, float]:
    """Convierte desplazamientos NED (north, east) a offsets (dlat, dlon) en grados."""
    lat0_rad = math.radians(lat0_deg)
    dlat_deg = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (east_m / (EARTH_RADIUS_M * math.cos(lat0_rad))) * (180.0 / math.pi)
    return dlat_deg, dlon_deg


def bearing_unit_vectors(bearing_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Devuelve vectores unitarios en NED:
      n_hat: desde la columna hacia el plano (bearing)
      l_hat: lateral izquierda (bearing + 90º)
    """
    br = math.radians(bearing_deg)
    n_hat = (math.cos(br), math.sin(br))            # (north, east)
    br_lat = br + math.pi / 2.0
    l_hat = (math.cos(br_lat), math.sin(br_lat))
    return n_hat, l_hat


def add_text_element(parent: ET.Element, tag: str, text: str) -> ET.Element:
    el = ET.SubElement(parent, tag)
    el.text = str(text)
    return el


def make_goto_step(lat: float, lon: float, depth: float,
                   altitude: float, surge_vel: float,
                   tol_xy: float, heave_mode: int = 0,
                   no_altitude_goes_up: bool = True) -> ET.Element:
    step = ET.Element("mission_step")
    man = ET.SubElement(step, "maneuver", {"type": "goto"})
    add_text_element(man, "final_latitude", f"{lat:.12f}")
    add_text_element(man, "final_longitude", f"{lon:.12f}")
    add_text_element(man, "final_depth", f"{depth:.2f}")
    add_text_element(man, "final_altitude", f"{altitude:.2f}")
    add_text_element(man, "heave_mode", f"{heave_mode}")
    add_text_element(man, "surge_velocity", f"{surge_vel:.3f}")
    add_text_element(man, "tolerance_xy", f"{tol_xy:.2f}")
    add_text_element(man, "no_altitude_goes_up", "true" if no_altitude_goes_up else "false")
    return step


def make_sway_section_step(lat_i: float, lon_i: float, depth_i: float,
                           lat_f: float, lon_f: float, depth_f: float,
                           altitude_f: float, sway_vel: float, tol_xy: float,
                           heave_mode: int = 0, no_altitude_goes_up: bool = True) -> ET.Element:
    """
    Crea una 'sway_section'
    """
    step = ET.Element("mission_step")
    man = ET.SubElement(step, "maneuver", {"type": "sway_section"})
    add_text_element(man, "initial_latitude", f"{lat_i:.12f}")
    add_text_element(man, "initial_longitude", f"{lon_i:.12f}")
    add_text_element(man, "initial_depth", f"{depth_i:.2f}")
    add_text_element(man, "final_latitude", f"{lat_f:.12f}")
    add_text_element(man, "final_longitude", f"{lon_f:.12f}")
    add_text_element(man, "final_depth", f"{depth_f:.2f}")
    add_text_element(man, "final_altitude", f"{altitude_f:.2f}")
    add_text_element(man, "heave_mode", f"{heave_mode}")
    add_text_element(man, "surge_velocity", f"{sway_vel:.3f}")
    add_text_element(man, "tolerance_xy", f"{tol_xy:.2f}")
    add_text_element(man, "no_altitude_goes_up", "true" if no_altitude_goes_up else "false")
    return step


def build_vertical_plane_mission(
                                 # --- MODIFICADO: Origen NED del Robot ---
                                 ned_origin_lat: float, ned_origin_lon: float,
                                 # --- MODIFICADO: Ubicación de la columna en metros ---
                                 column_north_m: float, column_east_m: float,
                                 column_radius_m: float, standoff_m: float,
                                 bearing_deg: float,
                                 plane_width_m: float,
                                 lateral_step_m: float,          # compat.
                                 top_depth_m: float, bottom_depth_m: float,
                                 vertical_step_m: float,          # paso entre bandas
                                 surge_velocity_mps: float = 0.4,
                                 tolerance_xy_m: float = 1.5,
                                 heave_mode: int = 0,
                                 altitude_stub: float = 0.0,
                                 start_on_left: bool = True) -> ET.ElementTree:
    """
    Genera la misión vertical NED con cobertura 'back-and-forth', usando
    SwaySection para que el AUV avance lateralmente.
    """
    assert bottom_depth_m > top_depth_m >= 0.0, "Profundidades incoherentes (bottom > top >= 0)."
    assert plane_width_m > 0 and vertical_step_m > 0, "Anchura y paso vertical deben ser > 0."

    root = ET.Element("mission")
    add_text_element(root, "version", "2.0")

    n_hat, l_hat = bearing_unit_vectors(bearing_deg)

    # --- LÓGICA MODIFICADA ---
    # 1. Centro del plano (a distancia r + standoff desde el eje de la columna)
    #    Calculado relativo al (0,0) NED.
    d = column_radius_m + standoff_m
    north_plane_center = column_north_m + n_hat[0] * d
    east_plane_center  = column_east_m  + n_hat[1] * d

    # 2. Bordes izquierdo (-W/2) y derecho (+W/2) del plano (en metros NED)
    half_w = plane_width_m / 2.0
    n_left  = north_plane_center + l_hat[0] * (-half_w)
    e_left  = east_plane_center  + l_hat[1] * (-half_w)
    n_right = north_plane_center + l_hat[0] * (+half_w)
    e_right = east_plane_center  + l_hat[1] * (+half_w)

    # 3. Convertir bordes (metros NED) a Lat/Lon absolutas usando el origen NED
    dlat_L, dlon_L = meters_to_latlon_offsets(ned_origin_lat, n_left,  e_left)
    dlat_R, dlon_R = meters_to_latlon_offsets(ned_origin_lat, n_right, e_right)
    lat_L, lon_L = ned_origin_lat + dlat_L, ned_origin_lon + dlon_L
    lat_R, lon_R = ned_origin_lat + dlat_R, ned_origin_lon + dlon_R
    # --- FIN LÓGICA MODIFICADA ---

    # Punto inicial: esquina superior izquierda (o derecha)
    lat_start, lon_start = (lat_L, lon_L) if start_on_left else (lat_R, lon_R)

    # 1) GOTO inicial a la esquina superior (lado elegido)
    root.append(
        make_goto_step(lat_start, lon_start, top_depth_m, altitude_stub,
                       surge_velocity_mps, tolerance_xy_m,
                       heave_mode, no_altitude_goes_up=True)
    )

    # 2) Cobertura back-and-forth con SwaySections
    current_depth = top_depth_m
    to_right = start_on_left  # True: L->R; False: R->L

    while current_depth < bottom_depth_m - 1e-6:
        if to_right:
            # Horizontal: Izquierda -> Derecha
            root.append(
                make_sway_section_step(lat_L, lon_L, current_depth,
                                       lat_R, lon_R, current_depth,
                                       altitude_stub, surge_velocity_mps,
                                       tolerance_xy_m, heave_mode,
                                       no_altitude_goes_up=True)
            )
            # Descenso vertical en el borde derecho
            next_depth = min(current_depth + vertical_step_m, bottom_depth_m)
            root.append(
                make_sway_section_step(lat_R, lon_R, current_depth,
                                       lat_R, lon_R, next_depth,
                                       altitude_stub, surge_velocity_mps,
                                       tolerance_xy_m, heave_mode,
                                       no_altitude_goes_up=True)
            )
        else:
            # Horizontal: Derecha -> Izquierda
            root.append(
                make_sway_section_step(lat_R, lon_R, current_depth,
                                       lat_L, lon_L, current_depth,
                                       altitude_stub, surge_velocity_mps,
                                       tolerance_xy_m, heave_mode,
                                       no_altitude_goes_up=True)
            )
            # Descenso vertical en el borde izquierdo
            next_depth = min(current_depth + vertical_step_m, bottom_depth_m)
            root.append(
                make_sway_section_step(lat_L, lon_L, current_depth,
                                       lat_L, lon_L, next_depth,
                                       altitude_stub, surge_velocity_mps,
                                       tolerance_xy_m, heave_mode,
                                       no_altitude_goes_up=True)
            )

        # Preparar siguiente banda
        current_depth = next_depth
        to_right = not to_right

    return ET.ElementTree(root)


def main():
    ap = argparse.ArgumentParser(description="Generador de misión vertical 'back-and-forth' (NED) usando SwaySection.")
    
    # --- ARGUMENTOS MODIFICADOS ---
    ap.add_argument("--ned_origin_lat", type=float, required=True, help="Latitud del origen NED del robot (deg).")
    ap.add_argument("--ned_origin_lon", type=float, required=True, help="Longitud del origen NED del robot (deg).")
    ap.add_argument("--column_north_m", type=float, required=True, help="Posición Norte de la columna (en metros) relativa al origen NED.")
    ap.add_argument("--column_east_m", type=float, required=True, help="Posición Este de la columna (en metros) relativa al origen NED.")
    # --- FIN ARGUMENTOS MODIFICADOS ---

    ap.add_argument("--column_radius_m", type=float, required=True, help="Radio de la columna (m).")
    ap.add_argument("--standoff_m", type=float, required=True, help="Distancia de seguridad (m).")
    ap.add_argument("--bearing_deg", type=float, required=True, help="Ángulo de la normal del plano (0=N, 90=E).")
    ap.add_argument("--plane_width_m", type=float, required=True, help="Anchura del plano (m).")
    ap.add_argument("--lateral_step_m", type=float, required=True, help="(Compat.) Ignorado en patrón back-and-forth.")
    ap.add_argument("--top_depth_m", type=float, required=True, help="Profundidad superior (m, positiva hacia abajo).")
    ap.add_argument("--bottom_depth_m", type=float, required=True, help="Profundidad inferior (m).")
    ap.add_argument("--vertical_step_m", type=float, required=True, help="Paso vertical entre pasadas (m).")
    ap.add_argument("--surge_velocity_mps", type=float, default=0.4, help="Velocidad deseada (m/s). Se usará como velocidad de SWAY.")
    ap.add_argument("--tolerance_xy_m", type=float, default=1.5, help="Tolerancia XY (m).")
    ap.add_argument("--heave_mode", type=int, default=0, help="Modo heave (0=profundidad).")
    ap.add_argument("--output", type=str, default="/home/rosuser/repo/catkin_ws/src/cola2_girona500/missions/mission_vertical_plane_NED.xml")
    args = ap.parse_args()

    tree = build_vertical_plane_mission(
        # --- ARGUMENTOS MODIFICADOS ---
        ned_origin_lat=args.ned_origin_lat,
        ned_origin_lon=args.ned_origin_lon,
        column_north_m=args.column_north_m,
        column_east_m=args.column_east_m,
        # --- FIN ARGUMENTOS MODIFICADOS ---

        column_radius_m=args.column_radius_m,
        standoff_m=args.standoff_m,
        bearing_deg=args.bearing_deg,
        plane_width_m=args.plane_width_m,
        lateral_step_m=args.lateral_step_m,
        top_depth_m=args.top_depth_m,
        bottom_depth_m=args.bottom_depth_m,
        vertical_step_m=args.vertical_step_m,
        surge_velocity_mps=args.surge_velocity_mps,
        tolerance_xy_m=args.tolerance_xy_m,
        heave_mode=args.heave_mode,
        altitude_stub=0.0,
        start_on_left=True
    )

    # Sangrado XML
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for e in elem:
                indent(e, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(tree.getroot())
    tree.write(args.output, encoding="utf-8", xml_declaration=False)
    print(f"Misión escrita en: {args.output}")


if __name__ == "__main__":
    main()