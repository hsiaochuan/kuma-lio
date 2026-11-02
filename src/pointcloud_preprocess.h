#ifndef FASTER_LIO_POINTCLOUD_PROCESSING_H
#define FASTER_LIO_POINTCLOUD_PROCESSING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>

#include <glog/logging.h>
#include "common_lib.h"

#include "laser_mapping_param.h"
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

namespace faster_lio {

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
    std::shared_ptr<LaserMappingParam> param;
};
}  // namespace faster_lio

#endif
