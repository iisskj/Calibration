cmake_minimum_required(VERSION 3.5)
#cmake_policy(SET CMP0074 NEW) 
project(RegionGrowingSegmentation)
set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_CXX_COMPILER "C:/AppProgram/mingw64/bin/g++")
# set(CMAKE_C_COMPILER "C:/AppProgram/mingw64/bin/gcc")

add_subdirectory(pybind11)

#set(PCL_DIR "C:/AppProgram/PCL 1.13.0/cmake")

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(segmentationext segmentation.cpp segmentation.hpp)
set_target_properties(segmentationext PROPERTIES PREFIX "")
set_target_properties(segmentationext PROPERTIES SUFFIX ".pyd")
target_link_libraries(segmentationext PRIVATE ${PCL_LIBRARIES})
