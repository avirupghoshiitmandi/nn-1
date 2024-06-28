cmake_minimum_required(VERSION 2.8.3)
project(runhoja)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
  CATKIN_DEPENDS rospy
)

install(PROGRAMS
  src/runhoja.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
