cmake_minimum_required(VERSION 2.8)
project(main_object-detection)

# The version number.
set (main_object-detection_VERSION_MAJOR 1)
set (main_object-detection_VERSION_MINOR 0)

SET( CMAKE_C_FLAGS   "-g -Wall -O3" )
SET( CMAKE_CXX_FLAGS "-Wall -g -O3" )
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # using regular Intel
    SET( CMAKE_CXX_FLAGS "-O3 -xHost -ipo -no-prec-div -static-intel -parallel -Wall" )
endif()



# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

if(NOT EXISTS "${opencv_object-detection_SOURCE_DIR}/bin/.gitignore")
    file(WRITE "${opencv_object-detection_SOURCE_DIR}/bin/.gitignore" "*\n")
endif()


#delete generated files in make clean
SET_DIRECTORY_PROPERTIES (
   PROPERTIES
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/*
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/bin/*
     ADDITIONAL_MAKE_CLEAN_FILES ${EXECUTABLE_OUTPUT_PATH}/*
)


# openCV library
set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
           CACHE PATH "The path where OpenCVConfig.cmake is placed")
FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( Boost REQUIRED COMPONENTS filesystem system regex program_options )
FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem program_options regex )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

ADD_LIBRARY( imreader SHARED imgreader.cpp imgreader.h )
target_link_libraries( imreader ${OpenCV_LIBS} ${Boost_LIBRARIES} )
set_property(TARGET imreader PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../lib)

ADD_EXECUTABLE( test_imread test_imreader.cpp )
TARGET_LINK_LIBRARIES( test_imread imreader )

# add the executable
ADD_EXECUTABLE( dnn_example opencv_dnn.cpp )
TARGET_LINK_LIBRARIES( dnn_example ${OpenCV_LIBS} )
set_property(TARGET dnn_example PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)

ADD_EXECUTABLE( selective selective.cpp )
TARGET_LINK_LIBRARIES( selective ${OpenCV_LIBS} )
set_property(TARGET selective PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)

ADD_EXECUTABLE( objdet objdet.cpp )
#TARGET_LINK_LIBRARIES( objdet ${OpenCV_LIBS} ${Boost_LIBRARIES} imreader )
TARGET_LINK_LIBRARIES( objdet ${OpenCV_LIBS} imreader ${Boost_PROGRAM_OPTIONS_LIBRARY} )
set_property(TARGET objdet PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)


INSTALL(
   FILES ${PROJECT_BINARY_DIR}/bin/obj_detection DESTINATION ${EXECUTABLE_OUTPUT_PATH}
   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
   )

INSTALL(
   FILES ${PROJECT_BINARY_DIR}/bin/selective DESTINATION ${EXECUTABLE_OUTPUT_PATH}
   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
)

INSTALL(
   FILES ${PROJECT_BINARY_DIR}/bin/objdet DESTINATION ${EXECUTABLE_OUTPUT_PATH}
   PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
)

