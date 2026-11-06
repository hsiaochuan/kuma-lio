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
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
add_definitions(-DPCL_NO_PRECOMPILE)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)
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
        ${GTSAM_INCLUDE_DIR}
        include
)


include_directories(src)
set(RTKLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/RTKLIB)
file(GLOB RTKLIB_SRCS CONFIGURE_DEPENDS ${RTKLIB_DIR}/src/*.c)
list(APPEND RTKLIB_SRCS
        ${RTKLIB_DIR}/src/rcv/binex.c
        ${RTKLIB_DIR}/src/rcv/crescent.c
        ${RTKLIB_DIR}/src/rcv/javad.c
        ${RTKLIB_DIR}/src/rcv/novatel.c
        ${RTKLIB_DIR}/src/rcv/nvs.c
        ${RTKLIB_DIR}/src/rcv/rt17.c
        ${RTKLIB_DIR}/src/rcv/septentrio.c
        ${RTKLIB_DIR}/src/rcv/skytraq.c
        ${RTKLIB_DIR}/src/rcv/swiftnav.c
        ${RTKLIB_DIR}/src/rcv/ublox.c
        ${RTKLIB_DIR}/src/rcv/unicore.c
)

add_library(rtklib STATIC ${RTKLIB_SRCS})
set_target_properties(rtklib PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
target_include_directories(rtklib PUBLIC ${RTKLIB_DIR}/src)
target_compile_definitions(rtklib PUBLIC
        ENAGLO ENAGAL ENAQZS ENACMP ENAIRN
        NFREQ=3 NEXOBS=3 SVR_REUSEADDR)

target_compile_options(rtklib PRIVATE -w)
target_link_libraries(rtklib PUBLIC m pthread)
foreach (rtklib_app rnx2rtkp convbin pos2kml)
        add_executable(${rtklib_app} ${RTKLIB_DIR}/app/consapp/${rtklib_app}/${rtklib_app}.c)
        target_link_libraries(${rtklib_app} rtklib)
        target_compile_options(${rtklib_app} PRIVATE -w)
endforeach ()
add_library(${PROJECT_NAME}
        src/laser_mapping.cc
        src/pointcloud_preprocess.cc
        src/pose3.cpp
        src/rgb_image.cpp
        src/stamp_pose.cpp
        src/voxel_map.cpp
        src/local_ba.cpp
        src/spline_state.cpp
        src/resple_lio.cpp
        src/fast_lio.cpp
        src/timer.cpp
        )

target_include_directories(${PROJECT_NAME} PRIVATE third_party/stb)
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        glog
        yaml-cpp
        gtsam
        rtklib
        gflags
        dw
        )

add_executable(run_mapping_offline app/run_mapping_offline.cc)
target_link_libraries(run_mapping_offline
        ${PROJECT_NAME}
        )

# pose graph optimization: its own library and executable, nothing here depends on the odometry
add_library(faster_pgo STATIC
        src/pgo/plane_cloud.cc
        src/pgo/icp_normal.cc
        src/pgo/btc.cc
        src/pgo/loop_detector.cc
        src/pgo/pose_graph.cc
        src/pgo/pgo_backend.cc
        src/stamp_pose.cpp
        src/timer.cpp
        )
# common_lib.h (VOXEL_LOCATION) includes the RTKLIB headers
target_link_libraries(faster_pgo
        rtklib
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenMP_CXX_LIBRARIES}
        glog
        gflags
        yaml-cpp
        gtsam
        )

add_executable(run_pgo app/run_pgo.cc)
target_link_libraries(run_pgo faster_pgo Boost::program_options)

add_executable(run_clean app/run_clean.cc)
target_link_libraries(run_clean ${PCL_LIBRARIES} ${OpenMP_CXX_LIBRARIES} glog Boost::program_options)

add_executable(run_ba app/run_ba.cc)
target_link_libraries(run_ba faster_pgo Boost::program_options)

add_executable(points_jet ./src/points_jet.cpp)
target_link_libraries(points_jet ${PROJECT_NAME})

add_executable(points_merge ./src/points_merge.cpp)
target_link_libraries(points_merge ${PROJECT_NAME})
