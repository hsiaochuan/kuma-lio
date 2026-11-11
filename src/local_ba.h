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

#include <array>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common_lib.h"
#include "point_cluster.h"
#include "stamp_pose.h"

namespace kuma_lio {

struct LbaParam {
    int win_size = 10;
    double voxel_size = 1.0;          // [m] layer-0 voxel; layers 1 and 2 halve it
    double plane_eigen_ratio = 0.1;   // plane when lambda_0 / lambda_1 is below this; scale free
    double min_plane_size = 0.1;      // smallest in-plane standard deviation a plane may have, as a
                                      // fraction of the cell edge of its own layer
    int clutter_points = 0;           // a cell holding this many points and lying on no plane is
                                      // dropped; 0 disables
    int min_points = 5;               // a node with this many points or fewer is neither a plane nor split
    int max_fix_points = 20;          // a finest cell stops absorbing marginalized points beyond this
    int max_iter = 3;                 // LM iterations per window
    double imu_coef = 2.5e-5;         // weight of the IMU cost against the LiDAR eigenvalue cost
    double cov_gyr = 1e-4;            // [(rad/s)^2] per gyroscope sample
    double cov_acc = 1.0;             // [(m/s^2)^2] per accelerometer sample
    double rdw_gyr = 1e-4;            // [(rad/s)^2 / s] gyroscope bias random walk (Voxel-SLAM rdw_gyr)
    double rdw_acc = 1e-4;            // [(m/s^2)^2 / s] accelerometer bias random walk (Voxel-SLAM rdw_acc)
    double sigma_num = 3.0;           // a plane explains a point within this many standard deviations
    double pose_var = 1e-4;           // [m^2] pose uncertainty added to every map point
    double range_err = 0.02;          // [m] LiDAR range error
    double angle_err = 0.05;          // [deg] LiDAR beam direction error
    double frame_dt = 0.1;            // [s] frame length
    double map_range = 200.0;         // [m] voxels farther than this from the sensor are dropped
    int num_threads = 4;              // for refitting the voxels
};


Mat3 BodyPointCov(const Vec3& p_body, double range_err, double angle_err);

// A point's match against the map.
struct PlaneHit {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 normal = Vec3::Zero();
    double d = 0.0;
    double signed_dist = 0.0;  // normal . p + d
    double plane_sigma = 0.0;  // variance of the signed distance caused by the plane's own uncertainty
};


struct LbaPoint {
    Vec3 p = Vec3::Zero();
    Mat3 var = Mat3::Zero();
};

struct LbaState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Mat3 R = Mat3::Identity();
    Vec3 p = Vec3::Zero();
    Vec3 v = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
};

// IMU preintegration between two frames (Voxel-SLAM IMU_PRE), integrated at the biases of the first
// frame and corrected to first order when the BA moves them.
class ImuPreint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat9 = Eigen::Matrix<double, 9, 9>;
    using Mat30 = Eigen::Matrix<double, 30, 30>;
    using Vec30 = Eigen::Matrix<double, 30, 1>;

    ImuPreint(const Vec3& bg, const Vec3& ba, const Eigen::Matrix<double, 6, 6>& noise_meas,
              const Eigen::Matrix<double, 6, 6>& noise_walk)
        : bg_(bg), ba_(ba), noise_meas_(noise_meas), noise_walk_(noise_walk) {}
    void Add(const Vec3& gyro, const Vec3& acc, double dt);
    // Residual (R p v bg ba) between the two frames, with the Gauss-Newton terms over both (R p v bg ba).
    double Evaluate(const LbaState& s1, const LbaState& s2, const Vec3& g, Mat30* jtj = nullptr,
                    Vec30* gg = nullptr) const;

   private:
    Vec3 bg_, ba_;
    Eigen::Matrix<double, 6, 6> noise_meas_, noise_walk_;
    Mat3 R_delta_ = Mat3::Identity();
    Vec3 p_delta_ = Vec3::Zero(), v_delta_ = Vec3::Zero();
    Mat3 R_bg_ = Mat3::Zero(), p_bg_ = Mat3::Zero(), p_ba_ = Mat3::Zero(), v_bg_ = Mat3::Zero(), v_ba_ = Mat3::Zero();
    Mat9 cov_ = Mat9::Zero();
    Eigen::Matrix<double, 6, 6> cov_walk_ = Eigen::Matrix<double, 6, 6>::Zero();
    double dtime_ = 0.0;
};

class LocalBA {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int kDim = 15;     // R p v bg ba
    static constexpr int kLayers = 3;   // fixed depth of the voxel grid
    static constexpr int kCells = 64;   // finest cells per layer-0 voxel, 4 per axis

    // A scan the window let go of (Voxel-SLAM ScanPose): what the loop closure builds its pose graph from.
    struct MargScan {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id = 0;
        int64_t time_ns = 0;
        LbaState x;                     // refined by the last BA it was part of
        std::vector<LbaPoint> points;   // body frame
    };

    explicit LocalBA(const LbaParam& param);
    void InitMap(const std::vector<Vec3>& points, double var);
    // World points, each with the isotropic variance `var`, into the map as "fix" points (Voxel-SLAM
    // keyframe_loading).  Their voxels are refit by the next Refresh, i.e. in the next AddFrame.
    void AddFixedPoints(const std::vector<Vec3>& points, double var);
    void AddImu(int64_t time_ns, const Vec3& gyro, const Vec3& acc);
    bool AddFrame(int64_t time_ns, const LbaState& odom, std::vector<LbaPoint> points, const Vec3& gravity);
    // The scan AddFrame marginalized, once; false when the last AddFrame marginalized nothing.
    bool TakeMarginalized(MargScan* scan);

    // Loop closure feedback (Voxel-SLAM loop_update).  ApplyCorrection moves every window state by the
    // rigid `dx` (R, p and v; the biases and gravity stay).  Rebuild throws the map away and builds it
    // again from `world_points` (variance `var`) and the window frames at their current poses.
    void ApplyCorrection(const Eigen::Isometry3d& dx);
    void Rebuild(const std::vector<Vec3>& world_points, double var);

    const LbaState& Newest() const { return frames_.back().x; }
    // The frames still in the window as MargScans, oldest first, at their current poses: at the end of a run
    // they go into the loop closure's graph after the scans TakeMarginalized handed out (LaserMapping::Finish).
    std::vector<MargScan> WindowScans() const;
    bool Query(const LbaPoint& pv, PlaneHit* hit) const;
    bool PlaneNormal(const Vec3& p, Vec3* normal) const;

   private:
    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id = 0;
        int64_t time_ns = 0;
        LbaState x;
        std::shared_ptr<ImuPreint> imu;
        std::vector<LbaPoint> points;  // body frame, kept for Rebuild and the marginalized scan
    };
    using Mat9 = Eigen::Matrix<double, 9, 9>;

    struct FrameCluster {
        int id = 0;
        PointCluster clu;
        Mat9 cov = Mat9::Zero();
    };

    struct Cell {
        int idx = 0;       // 16 ix + 4 iy + iz, each in [0, 4)
        PointCluster fix;
        Mat9 fix_cov = Mat9::Zero();
        std::vector<FrameCluster> clusters;
    };

    struct Plane {
        int layer = 0, node = 0;
        Vec3 center = Vec3::Zero();
        Vec3 normal = Vec3::Zero();
        double radius = 0.0;  // sqrt of the largest eigenvalue
        Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();  // (normal, center)
    };
    // A plane seen from the window: the LiDAR BA factor of Voxel-SLAM.
    struct PlaneFactor {
        PointCluster fix;
        std::vector<PointCluster> clus;  // by window slot, body frame
    };
    // A layer-0 voxel.
    struct Root {
        std::vector<Cell> cells;
        std::vector<Plane> planes;
        std::vector<PlaneFactor> factors;  // of `planes`, as of the last refit
        bool active = false;  // listed in active_
        bool dirty = false;   // took points since the last refit
    };
    // Per-node sums of one layer of one voxel, reused across voxels.
    struct RefitScratch {
        std::vector<PointCluster> win, fix, all;
        Eigen::aligned_vector<Mat9> cov;
    };

    static int NodeOf(int idx, int layer) {
        const int s = kLayers - 1 - layer;
        return (((idx >> 4) >> s) << (2 * layer)) | ((((idx >> 2) & 3) >> s) << layer) | ((idx & 3) >> s);
    }
    int CellIndex(const Vec3& pw, const VOXEL_LOCATION& loc) const {
        const Vec3 corner(loc.x * param_.voxel_size, loc.y * param_.voxel_size, loc.z * param_.voxel_size);
        const Eigen::Vector3i c = ((pw - corner) * (4.0 / param_.voxel_size)).cast<int>().cwiseMax(0).cwiseMin(3);
        return 16 * c.x() + 4 * c.y() + c.z();
    }
    Cell& CellAt(const Vec3& pw);
    void InsertFrame(const Frame& frame, const std::vector<LbaPoint>& points);
    void Refresh(bool dirty_only = false);

    void RefitRoot(Root* root, RefitScratch* scratch) const;
    void Marginalize(const Frame& frame);

    void Slide(const Vec3& center);
    bool MatchRoot(const VOXEL_LOCATION& loc, const LbaPoint& pv, PlaneHit* hit) const;
    double Evaluate(const std::vector<LbaState, Eigen::aligned_allocator<LbaState>>& xs, Eigen::MatrixXd* hess,
                    Eigen::VectorXd* jact) const;
    double EvaluateLidar(const std::vector<LbaState, Eigen::aligned_allocator<LbaState>>& xs, Eigen::MatrixXd* hess,
                         Eigen::VectorXd* jact) const;
    void Optimize();

    LbaParam param_;
    Vec3 g_;  // gravity acceleration, world frame
    Eigen::Matrix<double, 6, 6> noise_meas_, noise_walk_;
    std::deque<Imu> imu_buf_;
    std::deque<Frame, Eigen::aligned_allocator<Frame>> frames_;
    std::unordered_map<VOXEL_LOCATION, Root> map_;
    std::vector<Root*> active_;
    std::vector<const PlaneFactor*> factors_;

    MargScan marg_;
    bool has_marg_ = false;
};

}  // namespace kuma_lio
