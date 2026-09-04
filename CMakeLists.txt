cmake_minimum_required(VERSION 3.5)
project(faster_lio)

set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        std_msgs
        pcl_ros
        tf
        eigen_conversions
        cv_bridge
        velodyne_pointcloud
        velodyne_msgs
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
add_definitions(-DPCL_NO_PRECOMPILE)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${yaml-cpp_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        include
)


include_directories(src)
include_directories(third_party/RTKLIB/src)
file(GLOB_RECURSE RTKLIB_SOURCE
        "third_party/RTKLIB/src/*.c"
)
list(REMOVE_ITEM RTKLIB_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB/src/rcv/tersus.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB/src/rcv/comnav.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB/src/postpos.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB/src/convrnx.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB/src/download.c"
)
add_library(${PROJECT_NAME}
        src/laser_mapping.cc
        src/pointcloud_preprocess.cc
        src/pose3.cpp
        src/stamp_pose.cpp
        src/voxel_map.cpp
        src/spline_state.cpp
        src/resple_lio.cpp
        src/ikd-Tree/ikd_Tree.cpp
        src/timer.cpp
        src/gnss_processor.cpp
        ${RTKLIB_SOURCE}
        )

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        ${OpenMP_CXX_LIBRARIES}
        glog
        yaml-cpp
        dw
        gflags
        )

add_executable(run_mapping_offline app/run_mapping_offline.cc)
target_link_libraries(run_mapping_offline
        ${PROJECT_NAME}
        )

add_executable(points_jet ./src/points_jet.cpp)
target_link_libraries(points_jet ${PROJECT_NAME})

add_executable(points_merge ./src/points_merge.cpp)
target_link_libraries(points_merge ${PROJECT_NAME})

find_package(GTest REQUIRED)
enable_testing()
add_executable(camera_test src/camera_test.cpp)
target_link_libraries(camera_test
        ${PROJECT_NAME}
        GTest::GTest
        GTest::Main
)
add_test(NAME camera_test COMMAND camera_test)
