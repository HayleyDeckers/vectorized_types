cmake_minimum_required (VERSION 2.6)
project (vectorized_types)
# The version number.
set (vectorized_types_VERSION_MAJOR 1)
set (vectorized_types_VERSION_MINOR 0)

#install(TARGETS vectorized_types)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)
