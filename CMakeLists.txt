CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(PROJECT_ORG WangKeAlchemist)
SET(PROJECT_NAME vhip)
SET(PROJECT_DESCRIPTION "Prototype code for implementing an variable-vertical inverted pendulum in Crocoddyl.")
SET(PROJECT_URL https://github.com/${PROJECT_ORG}/Reduced_order_model_crocoddyl)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/test.cmake)
INCLUDE(cmake/python.cmake)

IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
ENDIF()

SETUP_PROJECT()

# Project configuration
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find-external/CppAD/" ${CMAKE_MODULE_PATH})

FINDPYTHON()
ADD_PROJECT_DEPENDENCY(crocoddyl)
ADD_PROJECT_DEPENDENCY(pinocchio)
ADD_PROJECT_DEPENDENCY(example-robot-data 3.8.0 REQUIRED PKG_CONFIG_REQUIRES "example-robot-data >= 3.8.0")

ADD_SUBDIRECTORY(python)
ADD_SUBDIRECTORY(examples)
# IF(BUILD_TESTING)
#   ADD_SUBDIRECTORY(unittest)
# ENDIF(BUILD_TESTING)