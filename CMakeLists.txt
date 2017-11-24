cmake_minimum_required(VERSION 2.8)
project(opencv_object-detection)

#MESSAGE("OpenCV_SOURCE_DIR:                   ${OpenCV_SOURCE_DIR}")
#MESSAGE("CMAKE_CURRENT_BINARY_DIR:            ${CMAKE_CURRENT_BINARY_DIR}")
#MESSAGE("CMAKE_CURRENT_SOURCE_DIR:            ${CMAKE_CURRENT_SOURCE_DIR}")
#MESSAGE("PROJECT_BINARY_DIR:                  ${PROJECT_BINARY_DIR}")
#MESSAGE("PROJECT_SOURCE_DIR:                  ${PROJECT_SOURCE_DIR}")
#MESSAGE("opencv_object-detection_CURRENT_BINARY_DIR:  ${opencv_object-detection_CURRENT_BINARY_DIR}")
#MESSAGE("opencv_object-detection_CURRENT_SOURCE_DIR:  ${opencv_object-detection_CURRENT_SOURCE_DIR}")
#MESSAGE("opencv_object-detection_BINARY_DIR:  ${opencv_object-detection_BINARY_DIR}")
#MESSAGE("opencv_object-detection_SOURCE_DIR:  ${opencv_object-detection_SOURCE_DIR}")


if(EXISTS "${opencv_object-detection_SOURCE_DIR}/build")
  if(NOT EXISTS "${opencv_object-detection_SOURCE_DIR}/build/.gitignore")
    file(WRITE "${opencv_object-detection_SOURCE_DIR}/build/.gitignore" "*\n")
  endif()
endif()


ADD_SUBDIRECTORY(main)

