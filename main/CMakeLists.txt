cmake_minimum_required(VERSION 2.8)
project(object-detection)

# The version number.
set (object-detection_VERSION_MAJOR 1)
set (object-detection_VERSION_MINOR 0)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)


#MESSAGE("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
#MESSAGE("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
#MESSAGE("RUNTIME_OUTPUT_DIRECTORY: ${RUNTIME_OUTPUT_DIRECTORY}")
#MESSAGE("EXECUTABLE_OUTPUT_PATH  : ${EXECUTABLE_OUTPUT_PATH}")
#MESSAGE("LIBRARY_OUTPUT_PATH     : ${LIBRARY_OUTPUT_PATH}")
#MESSAGE("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
#MESSAGE("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")
#MESSAGE("object-detection_SOURCE_DIR: ${object-detection_SOURCE_DIR}")
#MESSAGE("object-detection_BINARY_DIR: ${object-detection_BINARY_DIR}")

#delete generated files in make clean
SET_DIRECTORY_PROPERTIES (
   PROPERTIES
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/*
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/bin/*
     ADDITIONAL_MAKE_CLEAN_FILES ${EXECUTABLE_OUTPUT_PATH}/*
)

#/home/courses/student48/project/build/object-detection/src/bin/obj_detection
#CMAKE_SOURCE_DIR: /home/courses/student48/project
#CMAKE_BINARY_DIR: /home/courses/student48/project/bin
#RUNTIME_OUTPUT_DIRECTORY:
#EXECUTABLE_OUTPUT_PATH  : /home/courses/student48/project/bin
#LIBRARY_OUTPUT_PATH     : /home/courses/student48/project/bin
#PROJECT_SOURCE_DIR: /home/courses/student48/project/object-detection/src
#PROJECT_BINARY_DIR: /home/courses/student48/project/build/object-detection/src
#object-detection_SOURCE_DIR: /home/courses/student48/project/object-detection/src
#object-detection_BINARY_DIR: /home/courses/student48/project/build/object-detection/src



# openCV library
set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
           CACHE PATH "The path where OpenCVConfig.cmake is placed")
FIND_PACKAGE( OpenCV REQUIRED )

# add the executable
ADD_EXECUTABLE( obj_detection main.cpp )
TARGET_LINK_LIBRARIES( obj_detection ${OpenCV_LIBS} )
set_property(TARGET obj_detection PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)

INSTALL(
   FILES ${PROJECT_BINARY_DIR}/bin/obj_detection DESTINATION ${EXECUTABLE_OUTPUT_PATH}
   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
)

