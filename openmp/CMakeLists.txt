cmake_minimum_required(VERSION 2.8)
project(openmp_object-detection)

# The version number.
set (openmp_object-detection_VERSION_MAJOR 1)
set (openmp_object-detection_VERSION_MINOR 0)

FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem program_options regex )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )

# openCV library
set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
           CACHE PATH "The path where OpenCVConfig.cmake is placed")
FIND_PACKAGE( OpenCV REQUIRED )

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package(OpenMP)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_C_FLAGS "-fopenmp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

ADD_EXECUTABLE( objdet_openmp objdet_openmp.cpp )
TARGET_LINK_LIBRARIES( objdet_openmp ${OpenCV_LIBS} imreader googlenet ${Boost_PROGRAM_OPTIONS_LIBRARY} )
set_property(TARGET objdet_openmp PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)




# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../src")
MESSAGE("PROJECT_BINARY_DIR:                  ${PROJECT_BINARY_DIR}")
MESSAGE("PROJECT_SOURCE_DIR:                  ${PROJECT_SOURCE_DIR}")
#INCLUDE_DIRECTORIES($ENV{HOME}/local/include)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

#if(NOT EXISTS "${opencv_object-detection_SOURCE_DIR}/bin/.gitignore")
#    file(WRITE "${opencv_object-detection_SOURCE_DIR}/bin/.gitignore" "*\n")
#endif()


#delete generated files in make clean
SET_DIRECTORY_PROPERTIES (
   PROPERTIES
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/*
     ADDITIONAL_MAKE_CLEAN_FILES ${PROJECT_BINARY_DIR}/bin/*
     ADDITIONAL_MAKE_CLEAN_FILES ${EXECUTABLE_OUTPUT_PATH}/*
)


# openCV library
#set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
#           CACHE PATH "The path where OpenCVConfig.cmake is placed")
#FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( Boost REQUIRED COMPONENTS filesystem system regex program_options )


#ADD_EXECUTABLE( hello_mpi hello_mpi.cpp )
#TARGET_LINK_LIBRARIES( hello_mpi ${MPI_LIBRARIES} )
#if(MPI_COMPILE_FLAGS)
#  set_target_properties(hello_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()
#if(MPI_LINK_FLAGS)
#  set_target_properties(hello_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
#set_property(TARGET hello_mpi PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)


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

