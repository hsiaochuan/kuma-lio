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
using PointType = faster_lio::Point;
using PointCloud = pcl::PointCloud<faster_lio::Point>;
using PointVector = std::vector<faster_lio::Point, Eigen::aligned_allocator<faster_lio::Point>>;
namespace faster_lio {

constexpr double G_m_s2 = 9.81;  // Gravity const in GuangDong/China

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<double> &v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const boost::array<S, 3> &v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<double> &v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const boost::array<S, 9> &v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

template <typename S>
inline Eigen::Quaternion<S> QuatFromArray(const std::vector<double> &v) {
    Eigen::Quaternion<S> q;
    q.x() = v[0];
    q.y() = v[1];
    q.z() = v[2];
    q.w() = v[3];
    return q;
}
template <typename S>
inline Eigen::Matrix<S, 3, 3> RotationFromArray(const std::vector<double> &v) {
    if (v.size() != 9 && v.size() != 4) throw std::runtime_error("Invalid rotation matrix");
    Eigen::Matrix<S, 3, 3> rotation;
    if (v.size() == 9) rotation = MatFromArray<double>(v);
    else if (v.size() == 4)
        rotation = QuatFromArray<double>(v).toRotationMatrix();
    return rotation;
}
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

/// sync imu and lidar measurements
struct MeasureGroup {
    MeasureGroup() { this->lidar_.reset(new PointCloud()); };
    double lidar_end_time_ = 0;
    cv::Mat img_;
    PointCloud::Ptr lidar_ = nullptr;
    std::deque<Imu> imu_;
};
struct Pose6D {
    double offset_time = 0;
    boost::array<double, 3> acc;
    boost::array<double, 3> gyr;
    boost::array<double, 3> vel;
    boost::array<double, 3> pos;
    boost::array<double, 9> rot;
};

/**
 * set a pose 6d from ekf status
 * @tparam T
 * @param t
 * @param a
 * @param g
 * @param v
 * @param p
 * @param R
 * @return
 */
template <typename T>
Pose6D set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                  const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R) {
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return rot_kp;
}



template <typename T>
inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold = 0.1f) {
    if (point.size() < options::MIN_NUM_MATCH_POINTS) {
        return false;
    }

    Eigen::Matrix<T, 3, 1> normvec;

    if (point.size() == options::NUM_MATCH_POINTS) {
        Eigen::Matrix<T, options::NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, options::NUM_MATCH_POINTS, 1> b;

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < options::NUM_MATCH_POINTS; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        normvec = A.colPivHouseholderQr().solve(b);
    } else {
        Eigen::MatrixXd A(point.size(), 3);
        Eigen::VectorXd b(point.size(), 1);

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < point.size(); j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
        normvec(0, 0) = n(0, 0);
        normvec(1, 0) = n(1, 0);
        normvec(2, 0) = n(2, 0);
    }

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (const auto &p : point) {
        Eigen::Matrix<T, 4, 1> temp = p.getVector4fMap();
        temp[3] = 1.0;
        if (fabs(pca_result.dot(temp)) > threshold) {
            return false;
        }
    }
    return true;
}


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