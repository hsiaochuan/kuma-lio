/**
 * @file    lidar_simulator.hpp
 * @brief   Motion multi-line LiDAR simulator
 *
 * Features:
 *   - LiDAR moves along arbitrary trajectories (straight line / arc / figure-8 / custom)
 *   - Each laser point corresponds to an exact motion pose (per-point interpolation), automatically simulating motion
 * distortion
 *   - Supports distortion correction mode (project point cloud to frame-end pose)
 *   - Outputs timestamped motion point cloud, directly usable for SLAM / mapping algorithms
 *   - Supports CSV / PCD (PCL-compatible) / TUM trajectory format export
 *
 * Dependencies: Eigen3
 *   apt install libeigen3-dev
 *   Compile: g++ -O2 -std=c++17 -I/usr/include/eigen3 main.cpp -o lidar_sim
 *
 * Coordinate system (right-handed)
 *   X → forward  Y → left  Z → upward
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "common_lib.h"

namespace lidar_sim {
// ============================================================================
// Utility functions
// ============================================================================

static constexpr double kPi = M_PI;
static constexpr double kInf = std::numeric_limits<double>::infinity();

inline double deg2rad(double d) { return d * kPi / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / kPi; }

/** SE(3) pose interpolation: linear translation + SLERP rotation */
inline Eigen::Isometry3d interpolate_pose(const Eigen::Isometry3d &T0, const Eigen::Isometry3d &T1, double t) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = T0.translation() + t * (T1.translation() - T0.translation());
    T.linear() = Eigen::Quaterniond(T0.linear()).slerp(t, Eigen::Quaterniond(T1.linear())).toRotationMatrix();
    return T;
}

// ============================================================================
// LiDAR hardware configuration
// ============================================================================

struct LidarConfig {
    // Beam parameters
    int num_beams = 16;
    double vfov_min_deg = -15.0;
    double vfov_max_deg = +15.0;

    // Horizontal scan
    double h_res_deg = 0.2;
    double rotation_hz = 10.0;

    // Range
    double max_range = 30.0;
    double min_range = 0.05;

    // Noise (Gaussian)
    double range_noise_std = 0.02;

    // Derived quantities (filled by validate())
    int num_azimuth = 0;
    int total_points = 0;

    void validate() {
        if (num_beams < 1) throw std::invalid_argument("num_beams must be >= 1");
        if (h_res_deg <= 0) throw std::invalid_argument("h_res_deg must be > 0");
        if (max_range <= 0 || min_range >= max_range) throw std::invalid_argument("invalid range settings");
        if (vfov_min_deg >= vfov_max_deg) throw std::invalid_argument("vfov_min >= vfov_max");
        num_azimuth = static_cast<int>(std::round(360.0 / h_res_deg));
        total_points = num_beams * num_azimuth;
    }
};

// ============================================================================
// Timestamped pose & trajectory
// ============================================================================

struct StampedPose {
    double time;
    Eigen::Isometry3d pose;  ///< T_world_body

    StampedPose() : time(0), pose(Eigen::Isometry3d::Identity()) {}

    StampedPose(double t, const Eigen::Isometry3d &p) : time(t), pose(p) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using Trajectory = std::vector<StampedPose, Eigen::aligned_allocator<StampedPose> >;

// ============================================================================
// Trajectory generator
// ============================================================================

class TrajectoryGenerator {
   public:
    /**
     * Linear motion with constant velocity
     * @param start_pos  start position
     * @param velocity   velocity vector [m/s]
     * @param duration   total duration [s]
     * @param dt         sampling time step [s]
     */
    static Trajectory linear(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &velocity, double duration,
                             double dt = 0.01) {
        Trajectory traj;
        Eigen::Vector3d pos = start_pos;
        for (double t = 0; t <= duration + 1e-9; t += dt) {
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.translation() = pos;
            traj.push_back({t, T});
            pos += velocity * dt;
        }
        return traj;
    }

    /**
     * Circular motion with constant speed (horizontal plane)
     * @param center      circle center (x, y)
     * @param radius      radius [m]
     * @param linear_vel  linear velocity [m/s]
     * @param height      motion height z [m]
     * @param duration    total duration [s]
     */
    static Trajectory circular(const Eigen::Vector2d &center, double radius, double linear_vel, double height,
                               double duration, double dt = 0.01) {
        Trajectory traj;
        double w = linear_vel / radius;  // angular velocity
        double theta = 0.0;
        for (double t = 0; t <= duration + 1e-9; t += dt) {
            Eigen::Vector3d pos(center.x() + radius * std::cos(theta), center.y() + radius * std::sin(theta), height);
            double yaw = theta + kPi / 2.0;  // tangent direction
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.translation() = pos;
            T.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            traj.push_back({t, T});
            theta += w * dt;
        }
        return traj;
    }
};

// ============================================================================
// Trajectory interpolator (supports binary search + linear SLERP interpolation)
// ============================================================================

class TrajectoryInterpolator {
   public:
    explicit TrajectoryInterpolator(const Trajectory &traj) : traj_(traj) {
        if (traj_.empty()) throw std::invalid_argument("trajectory is empty");
    }

    double t_start() const { return traj_.front().time; }
    double t_end() const { return traj_.back().time; }

    Eigen::Isometry3d query(double t) const {
        if (t <= traj_.front().time) return traj_.front().pose;
        if (t >= traj_.back().time) return traj_.back().pose;

        // Binary search for adjacent frames
        std::size_t lo = 0, hi = traj_.size() - 1;
        while (hi - lo > 1) {
            std::size_t mid = (lo + hi) / 2;
            (traj_[mid].time <= t ? lo : hi) = mid;
        }
        double span = traj_[hi].time - traj_[lo].time;
        double frac = (span > 1e-12) ? (t - traj_[lo].time) / span : 0.0;
        return interpolate_pose(traj_[lo].pose, traj_[hi].pose, frac);
    }

    const Trajectory &trajectory() const { return traj_; }

   private:
    Trajectory traj_;
};

// ============================================================================
// Scene geometry (AABB)
// ============================================================================

struct SceneObject {
    Eigen::Vector3d min_pt, max_pt;
    double reflectivity = 0.8;
    std::string label;

    SceneObject() = default;

    SceneObject(const Eigen::Vector3d &mn, const Eigen::Vector3d &mx, double ref = 0.8, const std::string &lbl = "")
        : min_pt(mn), max_pt(mx), reflectivity(ref), label(lbl) {}

    // Slab method ray intersection
    double intersect(const Eigen::Vector3d &o, const Eigen::Vector3d &d, double t_max) const {
        double tmin = 1e-6, tmax = t_max;
        for (int i = 0; i < 3; ++i) {
            double inv = (std::abs(d[i]) > 1e-12) ? 1.0 / d[i] : kInf;
            double t1 = (min_pt[i] - o[i]) * inv;
            double t2 = (max_pt[i] - o[i]) * inv;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax + 1e-9) return kInf;
        }
        return (tmin < tmax) ? tmin : kInf;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class Scene {
   public:
    void add_object(const SceneObject &obj) { objects_.push_back(obj); }

    void add_room(double w, double d, double h, double wall_t = 0.1) {
        double hw = w / 2, hd = d / 2, t = wall_t;
        add_object({{-hw - t, -hd - t, -t}, {-hw, hd + t, h + t}, 0.6, "wall_L"});
        add_object({{hw, -hd - t, -t}, {hw + t, hd + t, h + t}, 0.6, "wall_R"});
        add_object({{-hw - t, -hd - t, -t}, {hw + t, -hd, h + t}, 0.6, "wall_F"});
        add_object({{-hw - t, hd, -t}, {hw + t, hd + t, h + t}, 0.6, "wall_B"});
        add_object({{-hw - t, -hd - t, -t}, {hw + t, hd + t, 0}, 0.5, "floor"});
        add_object({{-hw - t, -hd - t, h}, {hw + t, hd + t, h + t}, 0.4, "ceiling"});
    }

    double trace(const Eigen::Vector3d &origin, const Eigen::Vector3d &dir, double t_max,
                 double *ref_out = nullptr) const {
        double t_min = kInf, ref = 0;
        for (const auto &obj : objects_) {
            double t = obj.intersect(origin, dir, t_max);
            if (t < t_min) {
                t_min = t;
                ref = obj.reflectivity;
            }
        }
        if (ref_out) *ref_out = ref;
        return t_min;
    }

   private:
    std::vector<SceneObject> objects_;
};

// ============================================================================
// Single LiDAR point
// ============================================================================

struct LidarPoint {
    // Measurements
    int beam_id;
    int azimuth_idx;
    double azimuth_rad;
    double elevation_rad;
    double range;                 ///< Measured range with noise [m]
    double range_gt;              ///< Ground truth range without noise [m]
    double intensity;             ///< Reflectivity intensity [0,1]
    bool is_valid;                ///< Whether the ray hit an object
    double timestamp;             ///< Emission time of the point [s]
    Eigen::Vector3d point_lidar;  ///< LiDAR coordinate system

    faster_lio::Point Point() {
        faster_lio::Point point;
        point.getVector3fMap() = point_lidar.cast<float>();
        return point;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using LidarScan = std::vector<LidarPoint, Eigen::aligned_allocator<LidarPoint> >;

// ============================================================================
// LidarSimulator (motion-aware core class)
// ============================================================================

class LidarSimulator {
   public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @param cfg   LiDAR hardware configuration
     * @param traj  Vehicle motion trajectory (timestamped pose sequence)
     */
    LidarSimulator(LidarConfig cfg, const Trajectory &traj)
        : cfg_(std::move(cfg)),
          interp_(std::make_shared<TrajectoryInterpolator>(traj)),
          rng_(std::random_device{}()),
          noise_(0.0, 1.0) {
        cfg_.validate();
        build_tables();
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    Scene &scene() { return scene_; }
    const Scene &scene() const { return scene_; }
    const LidarConfig &config() const { return cfg_; }
    double t_start() const { return interp_->t_start(); }
    double t_end() const { return interp_->t_end(); }

    void set_seed(uint64_t s) { rng_.seed(s); }

    void set_trajectory(const Trajectory &traj) { interp_ = std::make_shared<TrajectoryInterpolator>(traj); }

    // ── Core scanning interface ──────────────────────────────────────────────

    /**
     * Scan one frame (LiDAR rotates one full revolution)
     *
     * Core mechanism:
     *   Each azimuth angle corresponds to a distinct time:
     *     pt_time = frame_start + (ai / num_azimuth) * (1 / rotation_hz)
     *   The vehicle pose at that time is obtained via trajectory interpolation,
     *   then combined with extrinsic to get the LiDAR pose.
     *   → The LiDAR moves during the scan, automatically producing motion distortion.
     *
     * @param frame_start_time   Start time of the frame rotation [s]
     * @param correct_distortion If true, transform all points to the frame-end reference pose
     *                           (simulates ideal distortion correction, for comparison)
     */
    LidarScan scan(double frame_start_time, bool correct_distortion = false) const {
        LidarScan cloud;
        cloud.reserve(cfg_.total_points);

        double frame_dur = 1.0 / cfg_.rotation_hz;

        // Reference pose for distortion correction (frame end)
        Eigen::Isometry3d T_world_scan_end = interp_->query(frame_start_time + frame_dur);

        for (int ai = 0; ai < cfg_.num_azimuth; ++ai) {
            double frac = static_cast<double>(ai) / cfg_.num_azimuth;
            double pt_time = frame_start_time + frac * frame_dur;

            Eigen::Isometry3d T_world_lpoint = interp_->query(pt_time);

            for (int bi = 0; bi < cfg_.num_beams; ++bi) {
                LidarPoint p = shoot_ray(bi, ai, azimuths_[ai], elevs_[bi], pt_time, T_world_lpoint);
                // Distortion correction: transform point from emission pose to reference pose
                if (correct_distortion && p.is_valid) {
                    p.point_lidar = T_world_scan_end.inverse() * T_world_lpoint * p.point_lidar;
                }
                cloud.push_back(std::move(p));
            }
        }
        return cloud;
    }

    std::vector<LidarScan> scan_sequence(double start_time, double end_time, bool correct_distortion = false) const {
        std::vector<LidarScan> result;
        double dt = 1.0 / cfg_.rotation_hz;
        double scan_time = start_time;
        for (; scan_time < end_time; scan_time += dt) result.push_back(scan(scan_time, correct_distortion));
        return result;
    }

    /**
     * Query LiDAR pose in world at any time
     */
    Eigen::Isometry3d query_lidar_pose(double t) const { return interp_->query(t); }

    /**
     * Get reference to the full trajectory (for export)
     */
    const Trajectory &trajectory() const { return interp_->trajectory(); }

    /** PCD ASCII (compatible with PCL / Open3D), world coordinates */
    static bool export_pcd(const LidarScan &cloud, const std::string &path) {
        int n = 0;
        for (const auto &p : cloud)
            if (p.is_valid) ++n;
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << "# .PCD v0.7 - lidar_sim\n"
          << "VERSION 0.7\n"
          << "FIELDS x y z intensity timestamp\n"
          << "SIZE 4 4 4 4 8\n"
          << "TYPE F F F F F\n"
          << "COUNT 1 1 1 1 1\n"
          << "WIDTH " << n << "\nHEIGHT 1\n"
          << "VIEWPOINT 0 0 0 1 0 0 0\n"
          << "POINTS " << n << "\nDATA ascii\n"
          << std::fixed << std::setprecision(6);
        for (const auto &p : cloud) {
            if (!p.is_valid) continue;
            f << p.point_lidar.x() << ' ' << p.point_lidar.y() << ' ' << p.point_lidar.z() << ' ' << p.intensity << ' '
              << p.timestamp << '\n';
        }
        return true;
    }

    /** Merge multiple frames into a map PCD */
    static bool export_map_pcd(const std::vector<LidarScan> &clouds, const Trajectory &stamped_poses,
                               const std::string &path) {
        int n = 0;
        for (const auto &c : clouds)
            for (const auto &p : c)
                if (p.is_valid) ++n;
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << "# .PCD v0.7 - lidar_sim map\n"
          << "VERSION 0.7\n"
          << "FIELDS x y z intensity timestamp\n"
          << "SIZE 4 4 4 4 8\nTYPE F F F F F\nCOUNT 1 1 1 1 1\n"
          << "WIDTH " << n << "\nHEIGHT 1\n"
          << "VIEWPOINT 0 0 0 1 0 0 0\n"
          << "POINTS " << n << "\nDATA ascii\n"
          << std::fixed << std::setprecision(6);
        for (int scan_i = 0; scan_i < clouds.size(); scan_i++) {
            const LidarScan &scan = clouds[scan_i];
            const auto lidar_pose = stamped_poses[scan_i];
            for (int pj = 0; pj < scan.size(); pj++) {
                if (!scan[pj].is_valid) continue;
                Eigen::Vector3d p_w = lidar_pose.pose * scan[pj].point_lidar;
                f << p_w.x() << ' ' << p_w.y() << ' ' << p_w.z() << ' ' << scan[pj].intensity << ' '
                  << scan[pj].timestamp << '\n';
            }
        }
        return true;
    }

    /**
     * Export trajectory in TUM format (compatible with evo evaluation tool)
     * Format: timestamp tx ty tz qx qy qz qw
     */
    static bool export_trajectory_tum(const Trajectory &traj, const std::string &path) {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << std::fixed << std::setprecision(9);
        for (const auto &sp : traj) {
            const auto &t = sp.pose.translation();
            Eigen::Quaterniond q(sp.pose.linear());
            f << sp.time << ' ' << t.x() << ' ' << t.y() << ' ' << t.z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z()
              << ' ' << q.w() << '\n';
        }
        return true;
    }

    // ── Point cloud filtering utilities ───────────────────────────────────────

    static LidarScan filter(const LidarScan &c, std::function<bool(const LidarPoint &)> pred) {
        LidarScan out;
        for (const auto &p : c)
            if (pred(p)) out.push_back(p);
        return out;
    }

   private:
    // ── Member variables ──────────────────────────────────────────────────────

    LidarConfig cfg_;
    Scene scene_;
    std::shared_ptr<TrajectoryInterpolator> interp_;
    mutable std::mt19937_64 rng_;
    mutable std::normal_distribution<double> noise_;

    std::vector<double> elevs_;     ///< Lookup table for elevation angles of each beam
    std::vector<double> azimuths_;  ///< Lookup table for azimuth angles

    // ── Build lookup tables ───────────────────────────────────────────────────

    void build_tables() {
        elevs_.resize(cfg_.num_beams);
        for (int i = 0; i < cfg_.num_beams; ++i) {
            double t = (cfg_.num_beams > 1) ? (double)i / (cfg_.num_beams - 1) : 0.5;
            elevs_[i] = deg2rad(cfg_.vfov_min_deg + t * (cfg_.vfov_max_deg - cfg_.vfov_min_deg));
        }
        azimuths_.resize(cfg_.num_azimuth);
        for (int i = 0; i < cfg_.num_azimuth; ++i) azimuths_[i] = deg2rad(i * cfg_.h_res_deg);
    }

    // ── Single ray emission ───────────────────────────────────────────────────

    LidarPoint shoot_ray(int beam_id, int az_idx, double az, double el, double timestamp,
                         const Eigen::Isometry3d &T_world_lidar) const {
        // Spherical coordinates → direction vector in LiDAR frame
        double cos_el = std::cos(el);
        Eigen::Vector3d dir_lidar(cos_el * std::cos(az), cos_el * std::sin(az), std::sin(el));

        // Transform to world frame
        Eigen::Vector3d origin = T_world_lidar.translation();
        Eigen::Vector3d dir_w = T_world_lidar.linear() * dir_lidar;

        // Ray tracing
        double ref = 0;
        double gt = scene_.trace(origin, dir_w, cfg_.max_range, &ref);

        LidarPoint p;
        p.beam_id = beam_id;
        p.azimuth_idx = az_idx;
        p.azimuth_rad = az;
        p.elevation_rad = el;
        p.timestamp = timestamp;
        p.is_valid = (gt < cfg_.max_range);
        p.range_gt = p.is_valid ? gt : cfg_.max_range;

        // Gaussian noise
        double meas = p.range_gt;
        if (p.is_valid && cfg_.range_noise_std > 0.0)
            meas = std::clamp(meas + cfg_.range_noise_std * noise_(rng_), cfg_.min_range, cfg_.max_range);
        p.range = meas;

        // Intensity (distance squared attenuation + material)
        p.intensity = p.is_valid ? std::clamp(ref / (1.0 + 0.005 * meas * meas), 0.0, 1.0) : 0.0;

        // Coordinates
        p.point_lidar = dir_lidar * p.range;
        return p;
    }
};

// ============================================================================
// Preset scene factories
// ============================================================================

namespace scene_factory {
/** Standard demo room (20m × 15m × 3m, with table, pillar, cylindrical obstacle) */
inline Scene make_demo_room() {
    Scene sc;
    sc.add_room(20.0, 20.0, 3.0);
    Eigen::Vector3d pillar1(5, 5, 0);
    Eigen::Vector3d pillar2(-5, -5, 0);
    Eigen::Vector3d pillar3(5, -5, 0);
    Eigen::Vector3d pillar_size(1.5, 1.5, 0);

    sc.add_object(
        {pillar1 - pillar_size / 2, pillar1 + pillar_size / 2 + Eigen::Vector3d::UnitZ() * 2.0, 0.8, "pillar1"});
    sc.add_object(
        {pillar2 - pillar_size / 2, pillar2 + pillar_size / 2 + Eigen::Vector3d::UnitZ() * 2.0, 0.8, "pillar2"});
    sc.add_object(
        {pillar3 - pillar_size / 2, pillar3 + pillar_size / 2 + Eigen::Vector3d::UnitZ() * 2.0, 0.8, "pillar3"});
    return sc;
}
}  // namespace scene_factory
}  // namespace lidar_sim
