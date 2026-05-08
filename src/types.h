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
const image_pair_t kInvalidImagePairId = std::numeric_limits<image_pair_t>::max();
const point2d_t kInvalidPoint2DIdx = std::numeric_limits<point2d_t>::max();
const landmark_t kInvalidPoint3DId = std::numeric_limits<landmark_t>::max();
const size_t kMaxNumImages = static_cast<size_t>(std::numeric_limits<int32_t>::max());
inline bool SwapImagePair(const image_t image_id1, const image_t image_id2) { return image_id1 > image_id2; }
inline image_pair_t ImagePairToPairId(const image_t image_id1, const image_t image_id2) {
    CHECK_LT(image_id1, kMaxNumImages);
    CHECK_LT(image_id2, kMaxNumImages);
    if (SwapImagePair(image_id1, image_id2)) {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id2 + image_id1;
    } else {
        return static_cast<image_pair_t>(kMaxNumImages) * image_id1 + image_id2;
    }
}

inline std::pair<image_t, image_t> PairIdToImagePair(const image_pair_t pair_id) {
    const image_t image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
    const image_t image_id1 = static_cast<image_t>((pair_id - image_id2) / kMaxNumImages);
    CHECK_LT(image_id1, kMaxNumImages);
    CHECK_LT(image_id2, kMaxNumImages);
    return std::make_pair(image_id1, image_id2);
}

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
    FeatureMatch(const point2d_t point2D_idx1, const point2d_t point2D_idx2)
        : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

    // Feature index in first image.
    point2d_t point2D_idx1 = kInvalidPoint2DIdx;

    // Feature index in second image.
    point2d_t point2D_idx2 = kInvalidPoint2DIdx;
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
struct Observation {
    Observation() = default;
    Observation(const image_t& image_id, const point2d_t& point2d_id) : image_id(image_id), point2d_id(point2d_id) {}
    // The image in which the track element is observed.
    image_t image_id;
    // The point in the image that the track element is observed.
    point2d_t point2d_id;
    bool operator==(const Observation& other) const {
        return (image_id == other.image_id && point2d_id == other.point2d_id);
    }
};
inline size_t ObservationToId(const Observation& obs) {
    return static_cast<size_t>(obs.image_id) << 32 | static_cast<size_t>(obs.point2d_id);
}
inline Observation IdToObservation(size_t id) {
    return Observation(static_cast<image_t>(id >> 32), static_cast<point2d_t>(id & 0xffffffff));
}
namespace std {
template <>
struct hash<Observation> {
    size_t operator()(const Observation& obs) const { return ObservationToId(obs); }
};
}  // namespace std
struct Landmark {
    Eigen::Vector3d xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    std::vector<Observation> track;
};