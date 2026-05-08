#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace faster_lio {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloud::Ptr &pcl_out, double scan_start) {
    LivoxHandler(msg, scan_start);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloud::Ptr &pcl_out, double scan_start) {
    switch (lidar_type_) {
        case LidarType::OUSTER:
            OusterHandler(msg, scan_start);
            break;
        case LidarType::HESAI:
            HesaiHandler(msg,scan_start);
            break;
        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg, double scan_start) {
    cloud_out_.clear();
    cloud_full_.clear();
    cloud_out_.reserve(msg->point_num);
    for (int i = 0; i < msg->point_num; i++) {
        if ((msg->points[i].tag & 0x30) != 0x10 && (msg->points[i].tag & 0x30) != 0x00)
            continue;
        if (i % point_filter_num_ != 0) continue;

        double range = msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y +
                       msg->points[i].z * msg->points[i].z;

        if (range < (blind_ * blind_)) continue;

        faster_lio::Point added_pt;
        added_pt.x = msg->points[i].x;
        added_pt.y = msg->points[i].y;
        added_pt.z = msg->points[i].z;
        added_pt.intensity = msg->points[i].reflectivity;
        // unit of offset_time: nanosecond
        added_pt.timestamp = scan_start + msg->points[i].offset_time / 1e9;

        cloud_out_.points.push_back(added_pt);
    }
}

void PointCloudPreprocess::OusterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start) {
    cloud_out_.clear();
    cloud_full_.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);


    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) continue;

        Eigen::Vector3d pt_vec;
        faster_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: nanosecond
        added_pt.timestamp = scan_start + pl_orig.points[i].t / 1e9;

        cloud_out_.points.push_back(added_pt);
    }
}
void PointCloudPreprocess::HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start) {
    cloud_out_.clear();
    pcl::PointCloud<hesai_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    cloud_out_.reserve(plsize);


    double base_time = msg->header.stamp.toSec();
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < blind_ * blind_) continue;

        faster_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        //
        added_pt.timestamp = scan_start + pl_orig.points[i].timestamp - base_time;
        cloud_out_.points.push_back(added_pt);
    }
}
}  // namespace faster_lio
