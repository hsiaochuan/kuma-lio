// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>

#include <glog/logging.h>
#include "common_lib.h"

namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (float, time, time)
                                (std::uint16_t, ring, ring)
)
// clang-format on

namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                (std::uint32_t, t, t)
                                (std::uint16_t, reflectivity, reflectivity)
                                (std::uint8_t, ring, ring)
                                (std::uint16_t, ambient, ambient)
                                (std::uint32_t, range, range)
)
// clang-format on

namespace hesai_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace hesai_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (double, timestamp, timestamp)
                                (std::uint16_t, ring, ring)
)
// clang-format on

namespace kuma_lio {

enum class LidarType {
    LIVOX = 1,
    OUSTER,
    HESAI,
    VELODYNE_POINTCLOUD2,
};
inline LidarType LidarTypeFromString(const std::string &lidar_type_str) {
    if (lidar_type_str == "LIVOX") {
        return LidarType::LIVOX;
    } else if (lidar_type_str == "OUSTER") {
        return LidarType::OUSTER;
    } else if (lidar_type_str == "HESAI") {
        return LidarType::HESAI;
    } else if (lidar_type_str == "VELODYNE_POINTCLOUD2") {
        return LidarType::VELODYNE_POINTCLOUD2;
    }else {
        throw std::runtime_error("unknown lidar type: " + lidar_type_str);
    }
}

// What the handlers read, filled from the config by the app that owns the preprocess.
struct PreprocessParam {
    int livox_tag_mask = 0x3F;  // Livox points are dropped when any of these tag bits is set
    int point_filter_num = 1;   // keep one point in point_filter_num
    double blind = 0.0;         // [m] closer returns are dropped
    float det_range_ = 100.0f;  // [m] farther returns are dropped
};

/**
 * point cloud preprocess
 * just unify the point format from livox/velodyne to PCL
 */
class PointCloudPreprocess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloudPreprocess() = default;
    ~PointCloudPreprocess() = default;

    PointCloud::Ptr LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg, std::int64_t scan_start);
    PointCloud::Ptr OusterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start);
    PointCloud::Ptr HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start);
    PointCloud::Ptr VelodynePointsHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, std::int64_t scan_start);
    PreprocessParam param;
};
}  // namespace kuma_lio
