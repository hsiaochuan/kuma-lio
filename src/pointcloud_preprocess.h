#ifndef FASTER_LIO_POINTCLOUD_PROCESSING_H
#define FASTER_LIO_POINTCLOUD_PROCESSING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>

#include <glog/logging.h>
#include "common_lib.h"

#include <velodyne_msgs/VelodyneScan.h>
#include <velodyne_pointcloud/pointcloudXYZIRT.h>
#include <velodyne_pointcloud/rawdata.h>

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
    VELODYNE_SCAN,
};
inline LidarType LidarTypeFromString(const std::string &lidar_type_str) {
    if (lidar_type_str == "LIVOX") {
        return LidarType::LIVOX;
    } else if (lidar_type_str == "OUSTER") {
        return LidarType::OUSTER;
    } else if (lidar_type_str == "HESAI") {
        return LidarType::HESAI;
    } else if (lidar_type_str == "VELODYNE_SCAN") {
        return LidarType::VELODYNE_SCAN;
    } else {
        LOG(ERROR) << "Unknown lidar type: " << lidar_type_str;
        return LidarType::LIVOX;
    }
}

/**
 * point cloud preprocess
 * just unify the point format from livox/velodyne to PCL
 */
class PointCloudPreprocess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloudPreprocess() {
        std::string calibration_fname = "/opt/ros/noetic/share/velodyne_pointcloud/params/32db.yaml";
        // VLP16, 32C, 32E, VLS128, 64E
        if (raw_data.setupOffline(calibration_fname, "32E", 200.0, 0.0)) {
            throw std::runtime_error("failed to setup offline raw data");
        }

    }
    ~PointCloudPreprocess() = default;

    /// processors
    void Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloud::Ptr &pcl_out, double scan_start);
    void Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloud::Ptr &pcl_out, double scan_start);
    void Process(const velodyne_msgs::VelodyneScan::ConstPtr &msg, PointCloud::Ptr &pcl_out, double scan_start);
    void Set(LidarType lid_type, double bld, int pfilt_num);

    // accessors
    double &Blind() { return blind_; }
    int &PointFilterNum() { return point_filter_num_; }
    LidarType GetLidarType() const { return lidar_type_; }
    void SetLidarType(LidarType lt) { lidar_type_ = lt; }

   private:
    void LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg, double scan_start);
    void OusterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start);
    void HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start);
    void VelodyneScanHandler(const velodyne_msgs::VelodyneScan::ConstPtr &msg, double scan_start);
    PointCloud cloud_full_, cloud_out_;
    velodyne_rawdata::RawData raw_data;
    LidarType lidar_type_ = LidarType::LIVOX;
    int point_filter_num_ = 1;
    double blind_ = 0.01;
};
}  // namespace faster_lio

#endif
