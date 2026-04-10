#pragma once
#include <boost/filesystem.hpp>
#include "cameras/cameras.h"
#include "pose3.h"
using image_t = uint32_t;
using camera_t = uint32_t;
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
using namespace faster_lio;
namespace fs = boost::filesystem;
struct Image {
    using Ptr = std::shared_ptr<Image>;
    Image(image_t image_id) : image_id_(image_id) {}
    image_t image_id_ = kInvalidImageId;
    camera_t camera_id_ = kInvalidCameraId;
    CameraBase::Ptr camera_ = nullptr;
    std::optional<Pose3> cam_from_world_;
    std::string name_;
    double timestamp_;
    camera_t CameraId() const {
        CHECK(camera_id_ != kInvalidCameraId);
        return camera_id_;
    }
    CameraBase::Ptr Camera() {
        CHECK_NOTNULL(camera_);
        return camera_;
    }
    Pose3 Pose() const {
        CHECK(cam_from_world_.has_value());
        return *cam_from_world_;
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
struct Reconstruction {
    void LoadFromImages(const std::string& image_dir) {
        boost::filesystem::path image_path_dir(image_dir);
        image_t image_id = 0;
        for (boost::filesystem::directory_iterator it(image_path_dir); it != boost::filesystem::directory_iterator();
             ++it) {
            if (!boost::filesystem::is_regular_file(*it)) continue;
            std::string extension = it->path().extension().string();
            if (extension != ".jpg" && extension != ".png") continue;
            images_[image_id] = std::make_shared<Image>(image_id);
            images_[image_id]->name_ = it->path().string();
            images_[image_id]->camera_id_ = 0;
            image_id++;
        }
    }
    std::unordered_map<camera_t, std::shared_ptr<CameraBase>> cameras_;
    std::unordered_map<camera_t, std::shared_ptr<Image>> images_;
};