cmake_minimum_required(VERSION 3.28)

project(Gaussian_renderer C CXX)

include(FetchContent)

set (CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# External dependencies

# GLM
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 8d1fd52e5ab5590e2c81768ace50c72bae28f2ed #refs/tags/1.0.3
)

FetchContent_MakeAvailable(glm)

# Libraries

add_library(core_lib)

target_sources(core_lib
  PRIVATE
    src/gaussian.cpp
    src/ppm.cpp
    src/projection.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS
      include
    FILES
      include/gaussian.hpp
      include/ppm.hpp
      include/projection.hpp
)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
  PRIVATE
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    core_lib
    glm::glm
)
