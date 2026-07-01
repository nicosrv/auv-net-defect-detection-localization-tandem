cmake_minimum_required(VERSION 3.0.2)
project(tandem)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  visualization_msgs
  dynamic_reconfigure
  cola2_msgs
  net_hole_detector
)

generate_dynamic_reconfigure_options(
  cfg/VerticalInspector.cfg
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    visualization_msgs
    dynamic_reconfigure
    cola2_msgs
    net_hole_detector
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/ri.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)