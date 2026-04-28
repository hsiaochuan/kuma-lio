#pragma once
#include "cameras/cameras.h"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
using namespace faster_lio;
typedef uint32_t camera_t;
typedef uint32_t image_t;
typedef uint64_t image_pair_t;
typedef uint32_t point2D_t;
typedef uint64_t point3D_t;
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
const image_pair_t kInvalidImagePairId = std::numeric_limits<image_pair_t>::max();
const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
const point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();

struct FeatureKeypoint {
    FeatureKeypoint();
    FeatureKeypoint(float x, float y);
    FeatureKeypoint(float x, float y, float scale, float orientation);
    FeatureKeypoint(float x, float y, float a11, float a12, float a21, float a22);
    float x;
    float y;

    // Affine shape of the feature.
    float a11;
    float a12;
    float a21;
    float a22;
};

inline FeatureKeypoint::FeatureKeypoint() : FeatureKeypoint(0, 0) {}

inline FeatureKeypoint::FeatureKeypoint(const float x, const float y) : FeatureKeypoint(x, y, 1, 0, 0, 1) {}

inline FeatureKeypoint::FeatureKeypoint(const float x_, const float y_, const float scale, const float orientation)
    : x(x_), y(y_) {
    CHECK_GE(scale, 0.0);
    const float scale_cos_orientation = scale * std::cos(orientation);
    const float scale_sin_orientation = scale * std::sin(orientation);
    a11 = scale_cos_orientation;
    a12 = -scale_sin_orientation;
    a21 = scale_sin_orientation;
    a22 = scale_cos_orientation;
}

inline FeatureKeypoint::FeatureKeypoint(const float x_, const float y_, const float a11_, const float a12_,
                                        const float a21_, const float a22_)
    : x(x_), y(y_), a11(a11_), a12(a12_), a21(a21_), a22(a22_) {}

typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptor;
typedef std::vector<FeatureKeypoint> FeatureKeypoints;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptors;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FeatureDescriptorsFloat;

struct FeatureMatch {
    FeatureMatch() : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx) {}
    FeatureMatch(const point2D_t point2D_idx1, const point2D_t point2D_idx2)
        : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

    // Feature index in first image.
    point2D_t point2D_idx1 = kInvalidPoint2DIdx;

    // Feature index in second image.
    point2D_t point2D_idx2 = kInvalidPoint2DIdx;
};

typedef std::vector<FeatureMatch> FeatureMatches;

struct TwoViewGeometry {
    // The configuration of the two-view geometry.
    enum ConfigurationType {
        UNDEFINED = 0,
        // Degenerate configuration (e.g., no overlap or not enough inliers).
        DEGENERATE = 1,
        // Essential matrix.
        CALIBRATED = 2,
        // Fundamental matrix.
        UNCALIBRATED = 3,
        // Homography, planar scene with baseline.
        PLANAR = 4,
        // Homography, pure rotation without baseline.
        PANORAMIC = 5,
        // Homography, planar or panoramic.
        PLANAR_OR_PANORAMIC = 6,
        // Watermark, pure 2D translation in image borders.
        WATERMARK = 7,
        // Multi-model configuration, i.e. the inlier matches result from multiple
        // individual, non-degenerate configurations.
        MULTIPLE = 8,
    };

    // One of `ConfigurationType`.
    int config = ConfigurationType::UNDEFINED;

    // Essential matrix.
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    // Fundamental matrix.
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    // Homography matrix.
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    // Relative pose.
    Pose3 cam2_from_cam1;

    // Inlier matches of the configuration.
    FeatureMatches inlier_matches;

    // Median triangulation angle.
    double tri_angle = -1;
};

struct Image {
    using Ptr = std::shared_ptr<Image>;
    image_t image_id_ = kInvalidImageId;
    camera_t camera_id_ = kInvalidCameraId;
    CameraBase::Ptr camera_ = nullptr;
    std::optional<Pose3> cam_from_world_;
    std::string name_;
    double timestamp_;
    cv::Mat image_data_;
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