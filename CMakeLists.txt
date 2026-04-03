cmake_minimum_required(VERSION 3.5)
project(faster_lio)

set(DEFAULT_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE release)
include(cmake/packages.cmake)

# options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)

if (WITH_IVOX_NODE_TYPE_PHC)
    message("USING_IVOX_NODE_TYPE_PHC")
    add_definitions(-DIVOX_NODE_TYPE_PHC)
else ()
    message("USING_IVOX_NODE_TYPE_DEFAULT")
endif()

add_library(${PROJECT_NAME}
        src/laser_mapping.cc
        src/pointcloud_preprocess.cc
        src/options.cc
        src/utils.cc
        src/voxel_map.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        tbb
        glog
        yaml-cpp
        )

include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
add_executable(run_mapping_online app/run_mapping_online.cc)
target_link_libraries(run_mapping_online
        ${PROJECT_NAME} gflags
        )
add_executable(run_mapping_offline app/run_mapping_offline.cc)
target_link_libraries(run_mapping_offline
        ${PROJECT_NAME} gflags
        )

find_package(GTest REQUIRED)
enable_testing()
add_executable(camera_test src/camera_test.cpp)
add_executable(lidar_sim_test src/lidar_sim_test.cc)
add_executable(bareg_test src/bareg_test.cc)
target_link_libraries(camera_test
        ${PROJECT_NAME}
        GTest::GTest
        GTest::Main
)
target_link_libraries(lidar_sim_test
        ${PROJECT_NAME}
        GTest::GTest
        GTest::Main
)
target_link_libraries(bareg_test
        ${PROJECT_NAME}
        GTest::GTest
        GTest::Main
)
add_test(NAME camera_test COMMAND camera_test)
add_test(NAME lidar_sim_test COMMAND lidar_sim_test)
add_test(NAME bareg_test COMMAND bareg_test)