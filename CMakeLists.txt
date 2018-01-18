cmake_minimum_required(VERSION 3.0)
PROJECT("cuda_dp")

#Set this with cmake, e.g. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0" CACHE STRING "Cuda root dir use \"-DCUDA_TOOLKIT_ROOT_DIR=...\" to set it")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(CUDA)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3" ) #It seems thse are passed to nvcc also...

# Pass options to NVCC
#-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30

#GeForce GTX 970 compute capability:	5.2
#-arch=sm_35 --ptxas-options=-v
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};        
    -arch=sm_30
     #-gencode=arch=compute_20,code=sm_20
     -gencode=arch=compute_30,code=sm_30
     -gencode=arch=compute_50,code=sm_50
     -gencode=arch=compute_52,code=sm_52
     -gencode=arch=compute_60,code=sm_60
     -gencode=arch=compute_61,code=sm_61
     -gencode=arch=compute_61,code=compute_61
    )

#Not needed for nvcc...
#include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_library(cu_speed_cost2go cu_speed_cost2go.cu  dp_utils.h SHARED)
target_link_libraries(cu_speed_cost2go ${CUDA_LIBRARIES})

add_executable(run_dp_cost2go run_dp_cost2go.cpp)
target_link_libraries(run_dp_cost2go cu_speed_cost2go)
