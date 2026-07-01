cmake_minimum_required(VERSION 3.0.2)
project(net_hole_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxArray.msg
  Detection3D.msg
  Detection3DArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/bbox_detector.py
  scripts/corrector_camera_info.py
  scripts/hole_segmenter_fastsam.py
  scripts/masked_stereo_image_builder.py
  scripts/stereo_distance_estimator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)