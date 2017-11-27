cmake_minimum_required(VERSION 2.8)
project(opencv_object-detection)

#MESSAGE("CMAKE_CURRENT_BINARY_DIR:            ${CMAKE_CURRENT_BINARY_DIR}")
#MESSAGE("CMAKE_CURRENT_SOURCE_DIR:            ${CMAKE_CURRENT_SOURCE_DIR}")
#MESSAGE("PROJECT_BINARY_DIR:                  ${PROJECT_BINARY_DIR}")
#MESSAGE("PROJECT_SOURCE_DIR:                  ${PROJECT_SOURCE_DIR}")
#MESSAGE("opencv_object-detection_CURRENT_BINARY_DIR:  ${opencv_object-detection_CURRENT_BINARY_DIR}")
#MESSAGE("opencv_object-detection_CURRENT_SOURCE_DIR:  ${opencv_object-detection_CURRENT_SOURCE_DIR}")
#MESSAGE("opencv_object-detection_BINARY_DIR:  ${opencv_object-detection_BINARY_DIR}")
#MESSAGE("opencv_object-detection_SOURCE_DIR:  ${opencv_object-detection_SOURCE_DIR}")

if(NOT EXISTS "${PROJECT_BINARY_DIR}/.gitignore")
    file(WRITE "${PROJECT_BINARY_DIR}/.gitignore" "*\n")
endif()

set(MODELS_DOWNLOAD_PATH "${PROJECT_SOURCE_DIR}/models")
if(NOT EXISTS "${MODELS_DOWNLOAD_PATH}")
    file(MAKE_DIRECTORY ${MODELS_DOWNLOAD_PATH})
endif()

if(NOT EXISTS "${MODELS_DOWNLOAD_PATH}/.gitignore")
    file(WRITE "${MODELS_DOWNLOAD_PATH}/.gitignore" "*\n")
endif()

file (COPY ${PROJECT_SOURCE_DIR}/src/download_models.sh 
    DESTINATION ${PROJECT_SOURCE_DIR}/models 
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ)

if(DOWNLOAD_CAFFE_MODELS)
    message("Downloading caffe models ...")
    set(download_cmd "${PROJECT_SOURCE_DIR}/models/download_models.sh")
    message(STATUS "download cmd: ${download_cmd}")
    execute_process(COMMAND ${download_cmd} 
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/models
        )
endif()

ADD_SUBDIRECTORY(src)

