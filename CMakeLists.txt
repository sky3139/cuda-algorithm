cmake_minimum_required(VERSION 1.6)
project(project)
# set(CMAKE_CXX_STANDARD 11)
find_package(CUDA)
include_directories ("${PROJECT_SOURCE_DIR}")

set(CUDA_NVCC_FLAGS -G;-g;-std=c++11)

# list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_61,code=sm_61")

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

# CUDA_ADD_EXECUTABLE(project ${CURRENT_HEADERS} ${CURRENT_SOURCES})
CUDA_ADD_EXECUTABLE(main  ./main.cu)
