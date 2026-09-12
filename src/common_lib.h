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
#include <opencv2/core/mat.hpp>

#include <pcl/io/pcd_io.h>
#include "pose3.h"
#include "so3_math.h"
#include "eigen_type.h"
#include "rtklib_.h"
#define S_TO_NS(x) (int64_t((x) * 1e9))
#define NS_TO_S(x) ((x) * 1e-9)
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
    std::uint32_t time_ns_h;
    std::uint32_t time_ns_l;
    std::uint16_t ring;
    // t_ns = (time_ns_h << 32) | time_ns_l
    inline std::int64_t GetTimeNs() const {
        std::uint64_t t_ns = (static_cast<std::uint64_t>(time_ns_h) << 32) |
                              static_cast<std::uint64_t>(time_ns_l);
        return static_cast<std::int64_t>(t_ns);
    }

    // time_ns_h / time_ns_l
    inline void SetTimeNs(std::int64_t t_ns) {
        std::uint64_t u = static_cast<std::uint64_t>(t_ns);
        time_ns_h = static_cast<std::uint32_t>(u >> 32);
        time_ns_l = static_cast<std::uint32_t>(u & 0xFFFFFFFFu);
    }

    // t_sec = t_ns * 1e-9
    inline double GetTimeSec() const {
        return static_cast<double>(GetTimeNs()) * 1e-9;
    }

    // t_ns = round(t_sec * 1e9)
    inline void SetTime(double t_sec) {
        std::int64_t t_ns = static_cast<std::int64_t>(std::llround(t_sec * 1e9));
        SetTime(t_ns);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 ColorPoint {
    PCL_ADD_POINT4D;
    float intensity;
    std::uint32_t time_ns_h;
    std::uint32_t time_ns_l;
    std::uint16_t ring;
    std::uint32_t rgb;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Imu {
    using Ptr = std::shared_ptr<Imu>;
    int64_t time_ns = -1;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
    Eigen::Vector3d orientation;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointWithNormal {
    Eigen::Vector3d xyz;
    Eigen::Vector3d normal;
};
}  // namespace faster_lio

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(faster_lio::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (std::uint32_t, time_ns_h, time_ns_h)
                                (std::uint32_t, time_ns_h, time_ns_l)
                                (std::uint16_t, ring, ring)
)
// clang-format on

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(faster_lio::ColorPoint,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (std::uint32_t, time_ns_h, time_ns_h)
                                (std::uint32_t, time_ns_h, time_ns_l)
                                (std::uint16_t, ring, ring)
                                (std::uint32_t, rgb, rgb)
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


using PointCloud = pcl::PointCloud<faster_lio::Point>;
using PointVector = std::vector<faster_lio::Point, Eigen::aligned_allocator<faster_lio::Point>>;
using ColorPointCloud = pcl::PointCloud<faster_lio::ColorPoint>;


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


struct RgbImage {
    using Ptr = std::shared_ptr<RgbImage>;
    std::string filename = std::string();
    std::int64_t time_ns_;
    cv::Mat image_data_;
};

struct MeasureGroup {
    MeasureGroup() { this->lidar_.reset(new PointCloud()); };
    PointCloud::Ptr lidar_ = nullptr;
    std::deque<Imu> imu_;
    std::vector<RgbImage> images;

    int obs_id = -1;
    int64_t obs_time_ns;
};

struct SatTrack {
    std::vector<int> obs_ids;
    std::vector<Eigen::Vector3d> sv_poss;
    std::vector<double> sv_dts;
    std::vector<int64_t> local_times;
    std::vector<double> phase_vars;  // this epoch's own undifferenced phase variance [m^2]
    double hatch = 0.0;

};

}  // namespace faster_lio
#endif