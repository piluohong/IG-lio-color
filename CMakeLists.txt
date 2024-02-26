cmake_minimum_required(VERSION 3.0.2)
project(ig_lio)

add_definitions(-w)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(TBB_DIR "/home/h/ku/TBB-2020_U2/cmake/templates")

# set(PCL_DIR "/home/h/ku/pcl-1.12.0/share/pcl-1.12")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  roslib
  pcl_ros
  livox_ros_driver2
  cv_bridge
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(PCL  REQUIRED)
find_package(TBB REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(OpenMP REQUIRED)

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL includepath: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL libpath: ${PCL_LIBRARY_DIRS}")

# include(cmake/glog.cmake)

# sophus
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sophus)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gicp_lio
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include 
)

# link includefile
include_directories(SYSTEM
  ${catkin_INCLUDE_DIRS}	
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#link .so
link_directories(${PCL_LIBRARY_DIRS} ${OpenMP_CXX_FLAGS}) 


add_executable(${PROJECT_NAME}_node 
  src/ig_lio_node
  src/pointcloud_preprocess.cpp
  src/lio.cpp
  src/voxel_map.cpp
  src/SymmetricEigenSolver.cpp
  src/timer.cpp
  src/faster_voxel_grid.cpp
)

# set_target_properties(${PROJECT_NAME}_node  PROPERTIES
#                       INSTALL_RPATH "/home/h/ku/pcl-1.12.0/lib")

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} 
${OpenCV_LIBRARIES}
${OpenMP_CXX_FLAGS}
  glog::glog
  gflags
  tbb
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  PATTERN ".svn" EXCLUDE)
