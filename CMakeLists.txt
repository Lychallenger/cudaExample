
cmake_minimum_required(VERSION 2.6)  #指定cmake最低版本为2.6

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70;-G;-g;-std=c++11;-Xcompiler;-fopenmp)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \
    --expt-extended-lambda
    ")
file(GLOB SOURCES
        *.cu
        *.cuh
        )
CUDA_ADD_EXECUTABLE(cuda ${SOURCES})