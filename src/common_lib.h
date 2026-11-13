// MIT License
//
// Copyright (c) 2026 hsiaochuan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <deque>
#include <string>
#include <vector>

#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include "pose3.h"
#include "so3_math.h"
#include "eigen_type.h"
#include "rtklib_.h"
#define S_TO_NS(x) (int64_t((x) * 1e9))
#define NS_TO_S(x) ((x) * 1e-9)
namespace kuma_lio {

extern double global_offset_time;
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
}  // namespace kuma_lio

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(kuma_lio::Point,
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
POINT_CLOUD_REGISTER_POINT_STRUCT(kuma_lio::ColorPoint,
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
struct hash<kuma_lio::VOXEL_LOCATION> {
    int64_t operator()(const kuma_lio::VOXEL_LOCATION &s) const {
        using std::hash;
        using std::size_t;
        return size_t(((s.x) * 73856093) ^ ((s.y) * 471943) ^ ((s.z) * 83492791)) % 10000000;
    }
};
}  // namespace std
namespace kuma_lio {


using PointCloud = pcl::PointCloud<kuma_lio::Point>;
using PointVector = std::vector<kuma_lio::Point, Eigen::aligned_allocator<kuma_lio::Point>>;
using ColorPointCloud = pcl::PointCloud<kuma_lio::ColorPoint>;


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
    int64_t time_ns = -1;
    int width = 0;   // of `data`
    int height = 0;
    std::vector<std::uint8_t> data;     // RGB8 pixels: a raw image, or a compressed one when image_decode is on
    std::vector<std::uint8_t> encoded;  // a CompressedImage's bytes, as received
    std::string format;                 // their format, "jpg" or "png", from CompressedImage::format

    bool Empty() const { return data.empty() && encoded.empty(); }
    bool Decode(const std::uint8_t *bytes, std::size_t size);
    bool SetEncoded(const std::uint8_t *bytes, std::size_t size, const std::string &msg_format);
    bool FromRaw(const std::uint8_t *pixels, int w, int h, int step, const std::string &encoding);
    // `format` for a compressed image, "png" for a raw one.
    std::string Extension() const { return encoded.empty() ? "png" : format; }
    // Writes <stem>.<Extension()>: the received bytes unchanged, or `data` as a PNG.
    bool Save(const std::string &stem) const;
};

struct MeasureGroup {
    MeasureGroup() { this->lidar_.reset(new PointCloud()); };
    PointCloud::Ptr lidar_ = nullptr;
    std::deque<Imu> imu_;
    RgbImage rgb_image_;
    int64_t mea_end_;
    int obs_id = -1;
    int64_t obs_time_ns;
};

struct SatTrack {
    std::vector<int> obs_ids;
    std::vector<Eigen::Vector3d> sv_poss;
    std::vector<double> sv_dts;
    std::vector<int64_t> local_times;
    double hatch = 0.0;

};

}  // namespace kuma_lio
