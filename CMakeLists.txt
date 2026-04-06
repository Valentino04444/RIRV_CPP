cmake_minimum_required(VERSION 3.18)
project(VideoTensorRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# --- OpenCV ---
find_package(OpenCV 4 REQUIRED)

# --- Modern CUDA (replaces the old FindCUDA) ---
find_package(CUDAToolkit REQUIRED)

# --- TensorRT (from apt install) ---
set(TENSORRT_ROOT "/usr")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include ${TENSORRT_ROOT}/src/tensorrt/include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found! Make sure tensorrt-dev is installed.")
endif()

# Include directories
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# Main executable
add_executable(video_inference 
    src/main.cpp
    src/preprocessing.cpp
    src/postprocessing.cpp
    src/renderer.cpp
)

# Link libraries
target_link_libraries(video_inference
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

# Status messages
message(STATUS "=== Build Configuration ===")
message(STATUS "TensorRT include : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library : ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV version   : ${OpenCV_VERSION}")
message(STATUS "CUDA Toolkit     : ${CUDAToolkit_VERSION}")
