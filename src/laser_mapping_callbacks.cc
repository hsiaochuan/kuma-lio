#include <cv_bridge/cv_bridge.h>

#include "laser_mapping.h"
#include "utils.h"

namespace faster_lio {
void LaserMapping::AddScanToBuffer(const PointCloud::Ptr &scan) {
    if (scan->empty()) {
        LOG(INFO) << "empty in scan, no points pushed into buffer";
        return;
    }
    std::sort(scan->points.begin(), scan->points.end(), [](const Point &p1, const Point &p2) { return p1.timestamp < p2.timestamp; });
    CHECK(scan->back().timestamp - scan->front().timestamp < 1.0) << "too big scan, check the timestamp";

    int skip_scan_points = 0;
    for (int i = 0; i < scan->size(); ++i) {
        if (points_buffer_.empty()) {
            points_buffer_.push_back(scan->at(i));
            continue;
        }
        if (points_buffer_.back().timestamp <= scan->at(i).timestamp) {
            points_buffer_.push_back(scan->at(i));
        }else {
            skip_scan_points++;
        }
    }
    if (skip_scan_points != 0)
        LOG(INFO) << "skip " << skip_scan_points << " scan points for timestamp of point error";
}
void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            double timestamp = msg->header.stamp.toSec();

            // time offset
            timestamp += param->lidar_time_offset_;

            // loop
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
            }
            last_timestamp_lidar_ = timestamp;

            // set start offset
            if (std::isnan(first_scan_time_)) {
                first_scan_time_ = timestamp;
            }

            timestamp = timestamp - first_scan_time_;

            // push to buffer
            PointCloud::Ptr scan(new PointCloud());
            switch (LidarTypeFromString(param->lidar_type)) {
                case LidarType::OUSTER:
                    scan = preprocess_->OusterHandler(msg, timestamp);
                    break;
                case LidarType::HESAI:
                    scan = preprocess_->HesaiHandler(msg, timestamp);
                    break;
                case LidarType::VELODYNE_POINTCLOUD2:
                    scan = preprocess_->VelodynePointsHandler(msg, timestamp);
                    break;
                default:
                    throw std::logic_error("unknown lidar type");
            }
            AddScanToBuffer(scan);
        },
        "Preprocess");
}

void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            double timestamp = msg->header.stamp.toSec();

            // time offset
            timestamp += param->lidar_time_offset_;

            // loop
            if (timestamp < last_timestamp_lidar_) {
                LOG(ERROR) << "lidar loop back, clear buffer";
            }
            last_timestamp_lidar_ = timestamp;

            // set start offset
            if (std::isnan(first_scan_time_)) {
                first_scan_time_ = timestamp;
            }

            timestamp = timestamp - first_scan_time_;

            // push to buffer
            PointCloud::Ptr scan;
            scan = preprocess_->LivoxHandler(msg, timestamp);
            AddScanToBuffer(scan);
        },
        "Preprocess");
}
void LaserMapping::VelodyneScanCallBack(const velodyne_msgs::VelodyneScan::ConstPtr &msg) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
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

            // set start offset
            if (std::isnan(first_scan_time_)) {
                first_scan_time_ = timestamp;
            }

            timestamp = timestamp - first_scan_time_;

            // push to buffer
            PointCloud::Ptr scan(new PointCloud());
            scan = preprocess_->VelodyneScanHandler(msg, timestamp);
            AddScanToBuffer(scan);
        },
        "Preprocess");
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

    // set start offset
    if (std::isnan(first_scan_time_)) {
        return;
    } else {
        timestamp = timestamp - first_scan_time_;
    }

    // push to buffer
    Imu imu;
    imu.timestamp = timestamp;
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
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        Image image;
        image.timestamp_ = msg_in->header.stamp.toSec();
        image.image_data_ = img;
        ImageCallBack(image);
    }
    img_count++;
}

void LaserMapping::ImageCallBack(Image &image) {
    // time offset
    image.timestamp_ += param->camera_time_offset_;

    // loop
    if (image.timestamp_ < last_timestamp_camera_) {
        LOG(WARNING) << "image loop back, clear buffer";
        image_buffer_.clear();
    }
    last_timestamp_camera_ = image.timestamp_;

    // set start offset
    if (std::isnan(first_scan_time_))
        return;
    else
        image.timestamp_ = image.timestamp_ - first_scan_time_;

    // push to buffer
    image_buffer_.emplace_back(image);
}

void LaserMapping::CompressedImageCallBack(const sensor_msgs::CompressedImage::ConstPtr &msg_in) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    static int img_count = 0;
    if (img_count % param->image_skip_ == 0) {
        cv::Mat img = cv_bridge::toCvCopy(msg_in, "bgr8")->image;
        Image image;
        image.timestamp_ = msg_in->header.stamp.toSec();
        image.image_data_ = img;
        ImageCallBack(image);
    }
    img_count++;
}

}  // namespace faster_lio

