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

include_directories(src)
add_library(${PROJECT_NAME}
        src/laser_mapping.cc
        src/laser_mapping_init.cc
        src/laser_mapping_callbacks.cc
        src/laser_mapping_publish.cc
        src/global_optimizor.cpp
        src/pointcloud_preprocess.cc
        src/options.cc
        src/imu_process.cpp
        src/pose3.cpp
        src/stamp_pose.cpp
        src/utils.cc
        src/voxel_map.cpp
        src/sfm_data.cpp
        src/database.cpp
        src/hash_octree.cpp
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
        sqlite3
        )

add_executable(run_mapping_online app/run_mapping_online.cc)
target_link_libraries(run_mapping_online
        ${PROJECT_NAME} gflags
        )
add_executable(run_mapping_offline app/run_mapping_offline.cc)
target_link_libraries(run_mapping_offline
        ${PROJECT_NAME} gflags
        )
add_executable(lidar_camera_calibration
        lidar_camera_calibration/lidar_camera_calib.cpp
        lidar_camera_calibration/lidar_camera_calib_impl.cpp
        lidar_camera_calibration/lidar_processor.cpp)
target_link_libraries(lidar_camera_calibration
        ${PROJECT_NAME} gflags
)
add_executable(points_jet ./src/points_jet.cpp)
target_link_libraries(points_jet ${PROJECT_NAME} gflags)

add_executable(points_merge ./src/points_merge.cpp)
target_link_libraries(points_merge ${PROJECT_NAME} gflags)

add_executable(lvba ./src/lvba.cpp)
target_link_libraries(lvba ${PROJECT_NAME} gflags)

add_executable(points_color ./src/points_color.cpp)
target_link_libraries(points_color ${PROJECT_NAME} gflags)

add_executable(offline_global_mapping ./src/offline_global_mapping.cpp)
target_link_libraries(offline_global_mapping ${PROJECT_NAME} gflags)

find_package(GTest REQUIRED)
enable_testing()
add_executable(camera_test src/camera_test.cpp)
add_executable(lidar_sim_test src/lidar_sim_test.cc)
add_executable(bareg_test src/bareg_test.cc)
add_executable(hash_octree_test src/hash_octree_test.cc)
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
target_link_libraries(hash_octree_test
        ${PROJECT_NAME}
        GTest::GTest
        GTest::Main
)
add_test(NAME camera_test COMMAND camera_test)
add_test(NAME lidar_sim_test COMMAND lidar_sim_test)
add_test(NAME bareg_test COMMAND bareg_test)
add_test(NAME hash_octree_test COMMAND hash_octree_test)
