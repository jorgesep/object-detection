cmake_minimum_required(VERSION 2.8)
project(utils_object-detection)

# The version number.
set (utils_object-detection_VERSION_MAJOR 1)
set (utils_object-detection_VERSION_MINOR 0)


SET( CMAKE_C_FLAGS   "-g -Wall -O3" )
SET( CMAKE_CXX_FLAGS "-Wall -g -O3" )
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # using regular Intel
    #SET( CMAKE_CXX_FLAGS "-O3 -xHost -ipo -no-prec-div -static-intel -parallel -Wall" )
    #SET( CMAKE_CXX_FLAGS "-O3 -ipo -axCORE-AVX2 -xSSE4.2")
    SET( CMAKE_CXX_FLAGS "-O3 -xHost -ipo")
endif()


#find_package(MPI REQUIRED)
#include_directories(${MPI_INCLUDE_PATH})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../src")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

#delete generated files in make clean
SET_DIRECTORY_PROPERTIES (
   PROPERTIES
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/*
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/bin/*
     ADDITIONAL_MAKE_CLEAN_FILES ${EXECUTABLE_OUTPUT_PATH}/*
)

#link_directories($ENV{HOME}/local/lib64)

# openCV library
set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
           CACHE PATH "The path where OpenCVConfig.cmake is placed")
FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem program_options regex )
#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

ADD_LIBRARY( utils SHARED utils.cpp utils.h )
#target_link_libraries( imreader ${OpenCV_LIBS} ${Boost_LIBRARIES} )
target_link_libraries( utils ${OpenCV_LIBS} )
set_property(TARGET utils PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../lib)

#INSTALL(
#   FILES ${PROJECT_BINARY_DIR}/bin/obj_detection DESTINATION ${EXECUTABLE_OUTPUT_PATH}
#   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
#   )
#
#INSTALL(
#   FILES ${PROJECT_BINARY_DIR}/bin/selective DESTINATION ${EXECUTABLE_OUTPUT_PATH}
#   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
#)
#
#INSTALL(
#   FILES ${PROJECT_BINARY_DIR}/bin/objdet DESTINATION ${EXECUTABLE_OUTPUT_PATH}
#   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
#)

