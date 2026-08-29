#include <cv_bridge/cv_bridge.h>

#include <unordered_map>

#include "laser_mapping.h"

#define SCAN_HZ 10
namespace faster_lio {
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr &scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }

    int skip_scan_points = 0;
    for (int i = 0; i < scan->size(); ++i) {
        if (points_buffer_.empty()) {
            points_buffer_.push_back(scan->at(i));
            continue;
        }
        if (points_buffer_.back().GetTimeNs() <= scan->at(i).GetTimeNs()) {
            points_buffer_.push_back(scan->at(i));
        } else {
            skip_scan_points++;
        }
    }
    if (skip_scan_points != 0) LOG(INFO) << "skip " << skip_scan_points << " scan points for timestamp of point error";
}
void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan(new PointCloud());
    switch (LidarTypeFromString(param->lidar_type)) {
        case LidarType::OUSTER:
            scan = preprocess_->OusterHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::HESAI:
            scan = preprocess_->HesaiHandler(msg, S_TO_NS(timestamp));
            break;
        case LidarType::VELODYNE_POINTCLOUD2:
            scan = preprocess_->VelodynePointsHandler(msg, S_TO_NS(timestamp));
            break;
        default:
            throw std::logic_error("unknown lidar type");
    }
    AddScanToBuffer(scan);
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    double timestamp = msg->header.stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) {
        return;
    }
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan;
    scan = preprocess_->LivoxHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}
void LaserMapping::VelodyneScanCallBack(const velodyne_msgs::VelodyneScan::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);

    CHECK(!msg->packets.empty()) << "VelodyneScan message is empty!";
    // the msg->header.stamp is 100 ms later than the msg->packets front
    double timestamp = msg->packets.front().stamp.toSec();

    // time offset
    timestamp += param->lidar_time_offset_;

    // loop
    if (timestamp < last_timestamp_lidar_) {
        LOG(ERROR) << "lidar loop back, clear buffer";
    }
    last_timestamp_lidar_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    PointCloud::Ptr scan(new PointCloud());
    scan = preprocess_->VelodyneScanHandler(msg, S_TO_NS(timestamp));
    AddScanToBuffer(scan);
}

void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    double timestamp = msg_in->header.stamp.toSec();

    // loop
    if (timestamp < last_timestamp_imu_) {
        LOG(WARNING) << "imu loop back, clear buffer";
        imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;
    if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
    timestamp = timestamp - global_offset_time;
    // push to buffer
    Imu imu;
    imu.time_ns = S_TO_NS(timestamp);
    imu.angular_velocity.x() = msg_in->angular_velocity.x;
    imu.angular_velocity.y() = msg_in->angular_velocity.y;
    imu.angular_velocity.z() = msg_in->angular_velocity.z;
    imu.linear_acceleration.x() = msg_in->linear_acceleration.x;
    imu.linear_acceleration.y() = msg_in->linear_acceleration.y;
    imu.linear_acceleration.z() = msg_in->linear_acceleration.z;
    imu_buffer_.emplace_back(imu);
}

void LaserMapping::ImageMsgCallBack(const sensor_msgs::Image::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (timestamp < last_timestamp_camera_) {
            LOG(WARNING) << "image loop back, clear buffer";
            image_buffer_.clear();
        }
        last_timestamp_camera_ = timestamp;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        Image image;
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        image.image_data_ = img;
        image.time_ns_ = S_TO_NS(timestamp);
        image_buffer_.emplace_back(image);
    }
    img_count++;
}

void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        double timestamp = msg_in->header.stamp.toSec();
        timestamp += param->camera_time_offset_;
        if (timestamp < last_timestamp_camera_) {
            LOG(WARNING) << "image loop back, clear buffer";
            image_buffer_.clear();
        }
        last_timestamp_camera_ = timestamp;
        if (std::isnan(global_offset_time) || timestamp < global_offset_time) return;
        timestamp = timestamp - global_offset_time;
        Image image;
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        image.image_data_ = img;
        image.time_ns_ = S_TO_NS(timestamp);
        image_buffer_.emplace_back(image);
    }
    img_count++;
}

}  // namespace faster_lio
