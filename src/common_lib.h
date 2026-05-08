#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <deque>
#include <string>
#include <vector>

#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/array.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include "options.h"
#include "pose3.h"
#include "so3_math.h"
#include <glog/logging.h>
namespace faster_lio {
class VOXEL_LOCATION {
   public:
    int64_t x;
    int64_t y;
    int64_t z;
    VOXEL_LOCATION(const Eigen::Vector3d &p, const double &v) {
        x = static_cast<int64_t>(std::floor(p(0) / v));
        y = static_cast<int64_t>(std::floor(p(1) / v));
        z = static_cast<int64_t>(std::floor(p(2) / v));
    }

    explicit VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }

    bool operator<(const VOXEL_LOCATION &b) const {
        if (x < b.x) return true;
        if (x > b.x) return false;

        if (y < b.y) return true;
        if (y > b.y) return false;

        return z < b.z;
    }
};
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 ColorPoint {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    std::uint16_t ring;
    uint32_t color;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline uint32_t RGBToU32(uint8_t r, uint8_t g, uint8_t b) {
    return (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
}
inline void U32ToRGB(uint32_t color, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = (color >> 16) & 0xFF;
    g = (color >> 8) & 0xFF;
    b = color & 0xFF;
}
struct Imu {
    using Ptr = std::shared_ptr<Imu>;
    double timestamp;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d orientation;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace faster_lio

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(faster_lio::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (double, timestamp, timestamp)
                                (std::uint16_t, ring, ring)
)
// clang-format on

namespace std {
template <>
struct hash<faster_lio::VOXEL_LOCATION> {
    int64_t operator()(const faster_lio::VOXEL_LOCATION &s) const {
        using std::hash;
        using std::size_t;
        return size_t(((s.x) * 73856093) ^ ((s.y) * 471943) ^ ((s.z) * 83492791)) % 10000000;
    }
};
}  // namespace std
namespace faster_lio {
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

using Vec2 = Eigen::Vector2d;
using Mat2X = Eigen::Matrix<double, 2, Eigen::Dynamic>;

using Vec3 = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat34 = Eigen::Matrix<double, 3, 4>;

using Vec4 = Eigen::Vector4d;
using Vec4f = Eigen::Vector4f;
using Mat4 = Eigen::Matrix<double, 4, 4>;

using PointCloud = pcl::PointCloud<faster_lio::Point>;
using PointVector = std::vector<faster_lio::Point, Eigen::aligned_allocator<faster_lio::Point>>;
using ColorPointCloud = pcl::PointCloud<faster_lio::ColorPoint>;

constexpr double G_m_s2 = 9.81;  // Gravity const in GuangDong/China

inline Vec3 VecFromArray(const std::vector<double> &v) {
    return Vec3(v[0], v[1], v[2]);
}

inline Mat3 MatFromArray(const std::vector<double> &v) {
    Mat3 m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

inline Eigen::Quaterniond QuatFromArray(const std::vector<double> &v) {
    Eigen::Quaterniond q;
    q.x() = v[0];
    q.y() = v[1];
    q.z() = v[2];
    q.w() = v[3];
    return q;
}

inline Mat3 RotationFromArray(const std::vector<double> &v) {
    if (v.size() != 9 && v.size() != 4) throw std::runtime_error("Invalid rotation matrix");
    Mat3 rotation;
    if (v.size() == 9) rotation = MatFromArray(v);
    else if (v.size() == 4)
        rotation = QuatFromArray(v).toRotationMatrix();
    return rotation;
}


/// sync imu and lidar measurements
struct MeasureGroup {
    MeasureGroup() { this->lidar_.reset(new PointCloud()); };
    double lidar_end_time_ = 0;
    cv::Mat img_;
    PointCloud::Ptr lidar_ = nullptr;
    std::deque<Imu> imu_;
};

using scan_t = uint32_t;
struct ScanFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<ScanFrame>;

    ScanFrame(const scan_t& scan_id) : scan_id(scan_id){}
    double GetTimestamp() const {
        CHECK(!std::isnan(timestamp));
        return timestamp;
    }
    double TryGetTimeFromName() {
        double image_stamp = 0.0;
        try {
            std::string image_stamp_str = boost::filesystem::path(cloud_fname).stem().string();
            image_stamp = std::stod(image_stamp_str);
        } catch (const std::exception &e) {
            throw std::runtime_error("fail to load the image timestamp from filename");
        }
        return image_stamp;
    }
    Pose3 GetWorldFromBody() const {
        CHECK(world_from_body.IsValid());
        return world_from_body;
    }
    PointCloud::Ptr GetScan() {
        if (scan) {
            return scan;
        }
        CHECK(!cloud_fname.empty());
        scan.reset(new PointCloud);
        pcl::io::loadPCDFile(cloud_fname, *scan);
        CHECK(!scan->empty());
        return scan;
    }

    scan_t scan_id = std::numeric_limits<scan_t>::max();
    std::string cloud_fname = std::string();
    Pose3 world_from_body = Pose3::InValid();
    double timestamp = std::numeric_limits<double>::quiet_NaN();
    PointCloud::Ptr scan = nullptr;
};
}  // namespace faster_lio
#endif