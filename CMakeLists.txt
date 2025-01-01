# cmake_minimum_required(VERSION ${CMAKE_VERSION})
# project(IntelFortranImplicit Fortran)
# add_custom_command(
#   OUTPUT output.cmake
#   COMMAND ${CMAKE_COMMAND} -P ${IntelFortranImplicit_SOURCE_DIR}/detect.cmake
#   )
# add_library(FortranLib hello.f output.cmake)

cmake_minimum_required(VERSION 3.10)

# Project name
project(Conv2DProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/Operators/include
    ${PROJECT_SOURCE_DIR}/Test_Operators/include
    ${PROJECT_SOURCE_DIR}/utilities/include
    ${PROJECT_SOURCE_DIR}/include
)


# Source files
file(GLOB_RECURSE OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Operators/src/*.cpp
)

file(GLOB_RECURSE TEST_OPERATORS_SRC
    ${PROJECT_SOURCE_DIR}/Test_Operators/src/*.cpp
)

file(GLOB_RECURSE UTILITIES_SRC
    ${PROJECT_SOURCE_DIR}/utilities/src/*.cpp
)

file(GLOB_RECURSE MAIN_SRC
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Add executable
add_executable(Conv2DApp
    ${OPERATORS_SRC}
    ${TEST_OPERATORS_SRC}
    ${UTILITIES_SRC}
    ${MAIN_SRC}
)


