#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace faster_lio {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

PointCloud::Ptr PointCloudPreprocess::LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                                                   double scan_start) {
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(msg->point_num);
    for (int i = 0; i < msg->point_num; i++) {
        if ((msg->points[i].tag & 0x30) != 0x10 && (msg->points[i].tag & 0x30) != 0x00) continue;

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

        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}

PointCloud::Ptr PointCloudPreprocess::OusterHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start) {
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
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

        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}

PointCloud::Ptr PointCloudPreprocess::VelodynePointsHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start) {
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) continue;
        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;
        if (range < (blind_ * blind_)) continue;
        faster_lio::Point added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: nanosecond
        added_pt.timestamp = scan_start + pl_orig.points[i].time;
        cloud_out->points.push_back(added_pt);
    }
    return cloud_out;
}
PointCloud::Ptr PointCloudPreprocess::HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, double scan_start) {
    pcl::PointCloud<hesai_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);

    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
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
        added_pt.timestamp = scan_start + pl_orig.points[i].timestamp - base_time;
    }
    return cloud_out;
}
PointCloud::Ptr PointCloudPreprocess::VelodyneScanHandler(const velodyne_msgs::VelodyneScan::ConstPtr &msg,
                                                          double scan_start) {
    raw_data.setParameters(0.0, 200.0, 0.0, 2.0 * M_PI);
    velodyne_pointcloud::PointcloudXYZIRT container(200.0, 0.0, "", "", raw_data.scansPerPacket());
    container.setup(msg);
    for (const auto &pkt : msg->packets) {
        raw_data.unpack(pkt, container, msg->packets.front().stamp);
    }
    const sensor_msgs::PointCloud2 &cloud_msg = container.finishCloud();
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(cloud_msg, pl_orig);

    PointCloud::Ptr cloud_out(new PointCloud);
    cloud_out->reserve(pl_orig.size());
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
        added_pt.timestamp = scan_start + pl_orig.points[i].time;
        cloud_out->points.push_back(added_pt);
    }
    cloud_out->width = cloud_out->points.size();
    cloud_out->height = 1;
    return cloud_out;
}
}  // namespace faster_lio
