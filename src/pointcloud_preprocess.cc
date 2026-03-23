#include "pointcloud_preprocess.h"

#include <glog/logging.h>
#include <execution>

namespace faster_lio {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloud::Ptr &pcl_out) {
    AviaHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloud::Ptr &pcl_out) {
    switch (lidar_type_) {
        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        case LidarType::HESAIxt32:
        case LidarType::ROBOSENSE:
            HesaiHandler(msg);
            break;

        case LidarType::LIVOX:
            LivoxHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type";
            break;
    }
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();
    int plsize = msg->point_num;

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, false);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从1开始
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        if ((msg->points[i].line < num_scans_) &&
            ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = msg->points[i].x;
                cloud_full_[i].y = msg->points[i].y;
                cloud_full_[i].z = msg->points[i].z;
                cloud_full_[i].intensity = msg->points[i].reflectivity;
                // unit of offset_time: nanosecond
                cloud_full_[i].time_nsec = msg->points[i].offset_time;

                if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
                    (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
                    (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                        (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                             cloud_full_[i].z * cloud_full_[i].z >
                         (blind_ * blind_))) {
                    is_valid_pt[i] = true;
                }
            }
        }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }
}

void PointCloudPreprocess::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
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
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: nanosecond
        added_pt.time_nsec = pl_orig.points[i].t;

        cloud_out_.points.push_back(added_pt);
    }
}

void PointCloudPreprocess::VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.SetTime(pl_orig.points[i].time * time_scale_);

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }
}

void PointCloudPreprocess::HesaiHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<hesai_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].timestamp > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    double time_head = pl_orig.points[0].timestamp;

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        // unit of offset_time: second
        added_pt.SetTime(pl_orig.points[i].timestamp - time_head);

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }
}

void PointCloudPreprocess::LivoxHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    cloud_out_.clear();
    cloud_full_.clear();

    pcl::PointCloud<livox_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, false);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;  // 从1开始
    }

    double timebase = pl_orig.points[0].timestamp;

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        if ((pl_orig.points[i].line < num_scans_) &&
            ((pl_orig.points[i].tag & 0x30) == 0x10 || (pl_orig.points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = pl_orig.points[i].x;
                cloud_full_[i].y = pl_orig.points[i].y;
                cloud_full_[i].z = pl_orig.points[i].z;
                cloud_full_[i].intensity = pl_orig.points[i].intensity;
                // unit of offset_time: nanosecond
                cloud_full_[i].time_nsec = pl_orig.points[i].timestamp - timebase;
                if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
                    (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
                    (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                        (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                             cloud_full_[i].z * cloud_full_[i].z >
                         (blind_ * blind_))) {
                    is_valid_pt[i] = true;
                }
            }
        }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }
}

}  // namespace faster_lio
