cmake_minimum_required (VERSION 2.6)
project (vectorized_types)
# The version number.
set (vectorized_types_VERSION_MAJOR 1)
set (vectorized_types_VERSION_MINOR 0)
#include all the flags when building the library so that it can select what it needs at compile time.
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f" )
file(GLOB SOURCES "src/*.cxx")
#Generate the static library from the sources
add_library(vectorized_types STATIC ${SOURCES})

#install(TARGETS vectorized_types)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)
install(TARGETS vectorized_types DESTINATION lib)
