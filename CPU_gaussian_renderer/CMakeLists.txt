cmake_minimum_required(VERSION 3.28)

project(Gaussian_renderer C CXX)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(core_lib)

target_sources(core_lib
  PRIVATE
    src/gaussian.cpp
    src/ppm.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      include
    FILES
      include/gaussian.hpp
      include/ppm.hpp
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
  PRIVATE
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    core_lib
)
