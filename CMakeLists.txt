cmake_minimum_required(VERSION 3.10) # cmake version
project(test) # project name
set(CMAKE_CXX_STANDARD 11) # c++11
set(CMAKE_BUILD_TYPE Debug) # Debug

# cuda
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) 
enable_language(CUDA)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/include/)
link_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/lib/)

# opnecv
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ./include)

include_directories(${PROJECT_SOURCE_DIR}/src/)
# Find recursively and store the file in ${SRCS}
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(client client.cpp) # executable
target_link_libraries(client ${OpenCV_LIBS} pthread) # lib

add_executable(server server.cpp ${SRCS}) # executable
target_link_libraries(server ${OpenCV_LIBS} pthread nvinfer cudart) # lib