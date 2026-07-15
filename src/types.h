#pragma once
#include <boost/filesystem.hpp>
#include "cameras/cameras.h"
namespace fs = boost::filesystem;
using namespace faster_lio;
typedef uint32_t camera_t;
typedef uint32_t image_t;
typedef uint64_t image_pair_t;
typedef uint32_t point2d_t;
typedef uint64_t landmark_t;
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();


struct Image {
    using Ptr = std::shared_ptr<Image>;
    image_t image_id_ = kInvalidImageId;
    camera_t camera_id_ = kInvalidCameraId;
    Pose3 cam_from_world_ = Pose3::InValid();
    std::string name_ = std::string();
    double timestamp_ = std::numeric_limits<double>::quiet_NaN();
    cv::Mat image_data_;

    std::vector<Eigen::Vector2d> points_;
    std::vector<landmark_t> landmark_ids_;
    camera_t CameraId() const {
        CHECK(camera_id_ != kInvalidCameraId);
        return camera_id_;
    }

    std::string Name() const {
        CHECK(!name_.empty());
        return name_;
    }

    double Timestamp() const {
        CHECK(std::isfinite(timestamp_));
        return timestamp_;
    }

    Pose3 CameraFromWorld() const {
        CHECK(cam_from_world_.IsValid());
        return cam_from_world_;
    }

    double TryReadTimeFromName() {
        double image_stamp = 0.0;
        try {
            std::string image_stamp_str = fs::path(name_).stem().string();
            image_stamp = std::stod(image_stamp_str);
        } catch (const std::exception& e) {
            throw std::runtime_error("fail to load the image timestamp from filename");
        }
        return image_stamp;
    }
};