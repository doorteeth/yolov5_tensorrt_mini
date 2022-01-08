cmake_minimum_required(VERSION 3.20)
project(yolov5mini_tensorrt)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(yolov5mini_tensorrt main.cpp)

target_include_directories(yolov5mini_tensorrt PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        )

target_link_libraries(yolov5mini_tensorrt PUBLIC
        ${OpenCV_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}

        nvinfer
        )