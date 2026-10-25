//
// Sliding-window LiDAR-inertial bundle adjustment and the local map it shares with the odometry,
// after Voxel-SLAM (Liu et al.): every plane voxel contributes the smallest eigenvalue of its point
// covariance, consecutive frames are tied by IMU preintegration, and the oldest frame is folded into
// the voxels as fixed points.  The planes are refitted from the window poses after every
// optimization, and the odometry matches its points against them.
//
// Kept minimal on purpose: single threaded, no map release, and a fixed three-layer grid (voxel_size, /2, /4) that is walked layer by layer instead of recursively.
//

#ifndef FASTER_LIO_LOCAL_BA_H
#define FASTER_LIO_LOCAL_BA_H
#include <array>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common_lib.h"
#include "point_cluster.h"
#include "stamp_pose.h"

namespace faster_lio {

struct LbaParam {
    int win_size = 10;
    double voxel_size = 1.0;          ///< [m] layer-0 voxel; layers 1 and 2 halve it
    double min_eigen_value = 0.0025;  ///< [m^2] plane when the smallest covariance eigenvalue is below this
    int min_points = 5;               ///< a node with this many points or fewer is neither a plane nor split
    int max_fix_points = 20;          ///< a finest cell stops absorbing marginalized points beyond this
    int max_iter = 3;                 ///< LM iterations per window
    double imu_coef = 2.5e-5;         ///< weight of the IMU cost against the LiDAR eigenvalue cost
    double cov_gyr = 1e-4;            ///< [(rad/s)^2] per gyroscope sample
    double cov_acc = 1.0;             ///< [(m/s^2)^2] per accelerometer sample
    double rdw_gyr = 1e-4;            ///< [(rad/s)^2 / s] gyroscope bias random walk (Voxel-SLAM rdw_gyr)
    double rdw_acc = 1e-4;            ///< [(m/s^2)^2 / s] accelerometer bias random walk (Voxel-SLAM rdw_acc)
    double sigma_num = 3.0;           ///< a plane explains a point within this many standard deviations
    double pose_var = 1e-4;           ///< [m^2] pose uncertainty added to every map point
    double range_err = 0.02;          ///< [m] LiDAR range error
    double angle_err = 0.05;          ///< [deg] LiDAR beam direction error
    bool cov_en = true;               ///< per-point residual variance from the plane and the point, else w_pt
    double r_base = 0.001;            ///< [m^2] added to the per-point residual variance when cov_en
    double fb_std_pos = 0.01;         ///< [m] noise of that pose measurement
    double fb_std_ort = 0.002;        ///< [rad]
    double fb_std_bg = 1e-3;          ///< [rad/s] noise of the BA gyroscope bias fed back with the pose
    double fb_std_ba = 1e-2;          ///< [m/s^2] noise of the BA accelerometer bias fed back with the pose
    bool fb_keep_cov = true;          ///< leave the filter covariance alone: the BA reuses the filter's own data
    double frame_dt = 0.1;            ///< [s] frame length
    double map_range = 200.0;         ///< [m] voxels farther than this from the sensor are dropped
    int num_threads = 4;              ///< for refitting the voxels
};


Mat3 BodyPointCov(const Vec3& p_body, double range_err, double angle_err);

/// A point's match against the map.
struct PlaneHit {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 normal = Vec3::Zero();
    double d = 0.0;
    double signed_dist = 0.0;  ///< normal . p + d
    double plane_sigma = 0.0;  ///< variance of the signed distance caused by the plane's own uncertainty
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

/// IMU preintegration between two frames (Voxel-SLAM IMU_PRE), integrated at the biases of the first
/// frame and corrected to first order when the BA moves them.
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
    /// Residual (R p v bg ba) between the two frames, with the Gauss-Newton terms over both (R p v bg ba).
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
    static constexpr int kDim = 15;     ///< R p v bg ba
    static constexpr int kLayers = 3;   ///< fixed depth of the voxel grid
    static constexpr int kCells = 64;   ///< finest cells per layer-0 voxel, 4 per axis

    explicit LocalBA(const LbaParam& param);
    void InitMap(const std::vector<Vec3>& points, double var);
    void AddImu(int64_t time_ns, const Vec3& gyro, const Vec3& acc);
    bool AddFrame(int64_t time_ns, const LbaState& odom, const std::vector<LbaPoint>& points, const Vec3& gravity);

    const LbaState& Newest() const { return frames_.back().x; }
    bool Query(const LbaPoint& pv, PlaneHit* hit) const;
    bool PlaneNormal(const Vec3& p, Vec3* normal) const;

   private:
    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id = 0;
        int64_t time_ns = 0;
        LbaState x;
        std::shared_ptr<ImuPreint> imu;  ///< from the previous frame to this one
    };
    using Mat9 = Eigen::Matrix<double, 9, 9>;

    struct WinPoints {
        int id = 0;       ///< frame id
        PointCluster pc;  ///< in that frame's body frame
        Mat9 cov = Mat9::Zero();  ///< Voxel-SLAM cov_add of these points, world frame at insertion
    };

    struct Cell {
        int idx = 0;       ///< 16 ix + 4 iy + iz, each in [0, 4)
        PointCluster fix;  ///< marginalized points, world frame
        Mat9 fix_cov = Mat9::Zero();
        std::vector<WinPoints> win;  ///< oldest frame first
    };

    struct Plane {
        int layer = 0, node = 0;
        Vec3 center = Vec3::Zero();
        Vec3 normal = Vec3::Zero();
        double radius = 0.0;  ///< sqrt of the largest eigenvalue
        Eigen::Matrix<double, 6, 6> plane_var = Eigen::Matrix<double, 6, 6>::Zero();  ///< (normal, center)
    };
    /// A plane seen from the window: the LiDAR BA factor of Voxel-SLAM.
    struct PlaneFactor {
        PointCluster fix;
        std::vector<PointCluster> win;  ///< by window slot, body frame
    };
    /// A layer-0 voxel.
    struct Root {
        std::vector<Cell> cells;
        std::vector<Plane> planes;
        std::vector<PlaneFactor> factors;  ///< of `planes`, as of the last refit
        bool active = false;  ///< listed in active_
        bool dirty = false;   ///< took points since the last refit
    };
    /// Per-node sums of one layer of one voxel, reused across voxels.
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
    Vec3 g_;  ///< gravity acceleration, world frame
    Eigen::Matrix<double, 6, 6> noise_meas_, noise_walk_;
    std::deque<Imu> imu_buf_;
    std::deque<Frame, Eigen::aligned_allocator<Frame>> frames_;
    std::unordered_map<VOXEL_LOCATION, Root> map_;
    std::vector<Root*> active_;
    std::vector<const PlaneFactor*> factors_;
};

}  // namespace faster_lio
#endif  // FASTER_LIO_LOCAL_BA_H
