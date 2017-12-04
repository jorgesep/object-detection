cmake_minimum_required(VERSION 2.8)
project(mpi_object-detection)

# The version number.
set (mpi_object-detection_VERSION_MAJOR 1)
set (mpi_object-detection_VERSION_MINOR 0)


find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem program_options regex )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIRS} )
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/../src")
include_directories("${PROJECT_SOURCE_DIR}/../utils")
# openCV library
set(OpenCV_DIR "$ENV{HOME}/local/share/OpenCV"
           CACHE PATH "The path where OpenCVConfig.cmake is placed")
FIND_PACKAGE( OpenCV REQUIRED )


ADD_EXECUTABLE( hello_mpi hello_mpi.cpp )
target_include_directories(hello_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(hello_mpi ${MPI_CXX_LIBRARIES})
set_property(TARGET hello_mpi PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)

ADD_EXECUTABLE( demo_mpi demo_mpi.cpp )
target_include_directories(demo_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(demo_mpi ${MPI_CXX_LIBRARIES})
set_property(TARGET demo_mpi PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)

ADD_EXECUTABLE( objdet_mpi objdet_mpi.cpp )
target_include_directories(objdet_mpi PRIVATE ${MPI_CXX_INCLUDE_PATH})
TARGET_LINK_LIBRARIES( objdet_mpi ${OpenCV_LIBS} utils imreader googlenet ${Boost_PROGRAM_OPTIONS_LIBRARY}  ${MPI_CXX_LIBRARIES})
set_property(TARGET objdet_mpi PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)


TARGET_LINK_LIBRARIES( hello_mpi ${MPI_LIBRARIES} )
#if(MPI_COMPILE_FLAGS)
#  set_target_properties(hello_mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()
#if(MPI_LINK_FLAGS)
#  set_target_properties(hello_mpi PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()



#set(MPI_C_COMPILER "mpicc")
#MPI_CXX_LIBRARIES and MPI_CXX_INCLUDE_PATH.
set_property(TARGET hello_mpi PROPERTY RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../../bin)


