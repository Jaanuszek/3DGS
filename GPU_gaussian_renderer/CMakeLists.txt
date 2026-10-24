cmake_minimum_required(VERSION 3.28)

project(Gaussian_renderer C CXX CUDA)

include(FetchContent)

# find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(HOST_LIB host_lib)
SET(DEVICE_LIB device_lib)

# GLM
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 8d1fd52e5ab5590e2c81768ace50c72bae28f2ed #refs/tags/1.0.3
)

FetchContent_MakeAvailable(glm)

# global DEBUG define for all targets builded with Debug config
add_compile_definitions(
  $<$<CONFIG:Debug>:DEBUG>
)

add_library(${HOST_LIB})

target_sources(${HOST_LIB}
  PRIVATE
    # src/ppm.cpp
    src/renderer.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      include
    FILES
      # include/ppm.hpp
      include/renderer.hpp
)

target_compile_options(${HOST_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-g;-O0>
)

add_library(${DEVICE_LIB})

target_sources(${DEVICE_LIB}
  PRIVATE
    cuda/ppm_cuda.cu
    cuda/renderer_cuda.cu
    cuda/gaussian.cu
    # cuda/helpers.cu
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      cuda
    FILES
      cuda/ppm_cuda.cuh
      cuda/renderer_cuda.cuh
      cuda/helpers.cuh
      cuda/gaussian.cuh
      cuda/cuda_check.cuh
)

target_compile_options(${DEVICE_LIB} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-G>
    # $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-g;-O0>
)

# https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html#separate-compilation-of-gpu-code
# it's needed because nvcc does not see the helpers.cu definitions
# By default nvcc expects whole-program compilation, and expects that all
# GPU code/symbols will be present in one unit during compilation
# So either inline tour __device__ function and create both declaration
# and definition in header file or set this:
# set_target_properties(
#   ${DEVICE_LIB}
#   PROPERTIES
#   CMAKE_CUDA_SEPARABLE_COMPILATION ON
#   )
# or -rdc = true
# It may introduce a small overhead due to linking stuff

add_executable(${PROJECT_NAME})

# target_compile_options(${PROJECT_NAME} PRIVATE -G -g -O0)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-G;-g>
    # $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-g;-O0>
)

target_sources(${PROJECT_NAME}
  PRIVATE
    main.cu
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${HOST_LIB}
    ${DEVICE_LIB}
    glm::glm
)
