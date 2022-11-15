cmake_minimum_required(VERSION 3.13)

project(myProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(ONNX_ML 1)

include_directories("/usr/local/cuda-11.8/targets/x86_64-linux/include/")
include_directories("./include/")
# include_directories("/usr/local/include")

# add_executable(myProject 
#     src/ioHelper.cpp
#     myProject.cpp
#     )


set_ifndef(PLUGINS_NEEDED ON) # ON
set_ifndef(SAMPLE_PARSERS "onnx") # "onnx"
set_ifndef(CUDA_INSTALL_DIR /usr/local/cuda)
set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(SAMPLES_COMMON_SOURCES
    ${SAMPLES_DIR}/common/logger.cpp
)
SET(SAMPLE_SOURCES
    myProject.cpp
)
add_executable(${TARGET_NAME}
    ${SAMPLE_SOURCES}
    ${SAMPLES_COMMON_SOURCES}
)

target_compile_definitions(myProject
    PRIVATE "ONNX_ML")

set(BUILD_PLUGINS 1)
set(BUILD_PARSERS 1)

set(DEPS_LIST "")
if(BUILD_PLUGINS)
    list(APPEND DEPS_LIST nvinfer_plugin)
endif()

if(BUILD_PARSERS)
    list(APPEND DEPS_LIST nvuffparser nvcaffeparser nvonnxparser)
endif()

if(BUILD_PLUGINS OR BUILD_PARSERS)
    add_dependencies(${TARGET_NAME}
        ${DEPS_LIST}
    )
endif()

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/parsers/onnx CACHE STRING "ONNX include directory")




