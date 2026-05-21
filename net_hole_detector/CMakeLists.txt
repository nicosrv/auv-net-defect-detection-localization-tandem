cmake_minimum_required(VERSION 3.0.2)
project(net_hole_detector)

# Busca las dependencias de ROS (rospy es la básica para Python)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  message_generation
)

# ------------------------------------------------
# 1. Configurar Librerías Python (setup.py)
# ------------------------------------------------
# Esto hace que 'net_hole_detector' sea importable desde cualquier lado
catkin_python_setup()

# Configurar Mensajes
add_message_files(
  FILES
  BoundingBox.msg      # <--- AÑADIR
  BoundingBoxArray.msg
  Detection3D.msg
  Detection3DArray.msg
  NetStats.msg
  AxisPoints.msg
  CoordinatesError.msg
)

add_service_files(
  FILES
  Trigger.srv
  CameraSelector.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)


# ------------------------------------------------
# 2. Configurar el Paquete ROS
# ------------------------------------------------
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES net_hole_detector
  CATKIN_DEPENDS rospy message_runtime std_msgs sensor_msgs geometry_msgs
#  DEPENDS system_lib
)


# ------------------------------------------------
# 3. Build
# ------------------------------------------------

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

# ------------------------------------------------
# 4. Instalar Ejecutables (Scripts)
# ------------------------------------------------
# Esto instala tu controlador para que puedas usar 'rosrun'
catkin_install_python(PROGRAMS
  scripts/detector_node.py
  scripts/manual_test_1.py
  scripts/blobs_processing.py
  scripts/hole_3d_pose_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# (Opcional) Si quieres instalar carpetas extra como launch o config
# install(DIRECTORY launch config
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )