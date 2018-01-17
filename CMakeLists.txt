cmake_minimum_required(VERSION 3.0)
PROJECT("cuda_dp")

#Set this with cmake, e.g. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0" CACHE STRING "Cuda root dir use \"-DCUDA_TOOLKIT_ROOT_DIR=...\" to set it")
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(CUDA)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3" ) #It seems thse are passed to nvcc also...

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_35 --ptxas-options=-v
    )

#Not needed for nvcc...
#include_directories(${CUDA_INCLUDE_DIRS})

#Simple example of toolchain...
#Note cuda_add_library takes "SHARED" arg last!!
cuda_add_library(simple simple.cu SHARED)
target_link_libraries(simple ${CUDA_LIBRARIES})

add_executable(simple_main run_simple.cpp)
target_link_libraries(simple_main simple)

#Dynamic programming for optimal control problem (cost-to-go style)
cuda_add_library(cu_dp_cost2go cu_dp_cost2go.cu SHARED)
target_link_libraries(cu_dp_cost2go ${CUDA_LIBRARIES})

cuda_add_library(cu_speed_cost2go cu_speed_cost2go.cu SHARED)
target_link_libraries(cu_speed_cost2go ${CUDA_LIBRARIES})

add_executable(run_dp_cost2go run_dp_cost2go.cpp)
target_link_libraries(run_dp_cost2go cu_speed_cost2go)
