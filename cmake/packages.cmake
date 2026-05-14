list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# glog
find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

# for ubuntu 18.04, update gcc/g++ to 9, and download tbb2018 from
# https://github.com/oneapi-src/oneTBB/releases/download/2018/tbb2018_20170726oss_lin.tgz,
# extract it into CUSTOM_TBB_DIR 
# specifiy tbb2018, e.g. CUSTOM_TBB_DIR=/home/idriver/Documents/tbb2018_20170726oss
if (CUSTOM_TBB_DIR)
    set(TBB2018_INCLUDE_DIR "${CUSTOM_TBB_DIR}/include")
    set(TBB2018_LIBRARY_DIR "${CUSTOM_TBB_DIR}/lib/intel64/gcc4.7")
    include_directories(${TBB2018_INCLUDE_DIR})
    link_directories(${TBB2018_LIBRARY_DIR})
endif ()

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
