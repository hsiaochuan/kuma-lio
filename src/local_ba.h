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

#include <opencv2/core.hpp>

#include "cameras/cameras.h"
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
    double sigma_num = 3.0;           ///< a plane explains a point within this many standard deviations
    double pose_var = 1e-4;           ///< [m^2] pose uncertainty added to every map point
    double range_err = 0.02;          ///< [m] LiDAR range error
    double angle_err = 0.05;          ///< [deg] LiDAR beam direction error
    bool cov_en = true;               ///< per-point residual variance from the plane and the point, else w_pt
    double r_base = 0.001;            ///< [m^2] added to the per-point residual variance when cov_en
    double fb_std_pos = 0.01;         ///< [m] noise of that pose measurement
    double fb_std_ort = 0.002;        ///< [rad]
    bool fb_keep_cov = true;          ///< leave the filter covariance alone: the BA reuses the filter's own data
    double frame_dt = 0.1;            ///< [s] frame length when there is no camera whose stamps cut the frames
    double map_range = 200.0;         ///< [m] voxels farther than this from the sensor are dropped
    int num_threads = 4;              ///< for refitting the voxels
    // DSO-style photometric factors between the window's images, with the LiDAR supplying the depth
    bool photo_enable = false;
    double photo_coef = 1e-6;         ///< weight of the photometric cost [gray^2] against the LiDAR eigenvalue cost
    int photo_grid = 24;              ///< [px] at most one point per image cell of this size
    double photo_grad_min = 15.0;     ///< [gray/px] smallest image gradient a point is picked at
    double photo_huber = 9.0;         ///< [gray] Huber threshold per pixel (DSO)
    double photo_outlier = 20.0;      ///< [gray] a patch with a mean error beyond this is ignored
    double photo_grad_c = 0.0;        ///< [gray/px] DSO's gradient weight c^2 / (c^2 + |grad I|^2) per pixel; 0 turns it off
    double photo_max_depth = 30.0;    ///< [m]
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
    Vec3 bg = Vec3::Zero();  ///< the biases are the filter's and are not optimized: one second of
    Vec3 ba = Vec3::Zero();  ///< window cannot tell a bias from motion the LiDAR leaves unobserved
};

/// IMU preintegration between two frames (Voxel-SLAM IMU_PRE) at fixed biases.
class ImuPreint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Mat9 = Eigen::Matrix<double, 9, 9>;
    using Mat18 = Eigen::Matrix<double, 18, 18>;
    using Vec18 = Eigen::Matrix<double, 18, 1>;

    ImuPreint(const Vec3& bg, const Vec3& ba, const Eigen::Matrix<double, 6, 6>& noise_meas)
        : bg_(bg), ba_(ba), noise_meas_(noise_meas) {}
    void Add(const Vec3& gyro, const Vec3& acc, double dt);
    double Evaluate(const LbaState& s1, const LbaState& s2, const Vec3& g, Mat18* jtj = nullptr,
                    Vec18* gg = nullptr) const;

   private:
    Vec3 bg_, ba_;
    Eigen::Matrix<double, 6, 6> noise_meas_;
    Mat3 R_delta_ = Mat3::Identity();
    Vec3 p_delta_ = Vec3::Zero(), v_delta_ = Vec3::Zero();
    Mat9 cov_ = Mat9::Zero();
    double dtime_ = 0.0;
};

class LocalBA {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    static constexpr int kDim = 9;      ///< R p v
    static constexpr int kLayers = 3;   ///< fixed depth of the voxel grid
    static constexpr int kCells = 64;   ///< finest cells per layer-0 voxel, 4 per axis

    explicit LocalBA(const LbaParam& param);
    void InitMap(const std::vector<Vec3>& points, double var);
    void AddImu(int64_t time_ns, const Vec3& gyro, const Vec3& acc);
    bool AddFrame(int64_t time_ns, const LbaState& odom, const std::vector<LbaPoint>& points, const Vec3& gravity,
                  const cv::Mat& gray = cv::Mat());
    void SetCamera(const CamModel::Ptr& cam, const Pose3& body_from_cam) {
        cam_ = cam;
        R_ic_ = body_from_cam.q_.toRotationMatrix();
        t_ic_ = body_from_cam.t_;
    }

    const LbaState& Newest() const { return newest_; }
    void SetOdomPose(const Eigen::Isometry3d& pose) { history_.back().odom = pose; }
    bool Query(const LbaPoint& pv, PlaneHit* hit) const;
    bool PlaneNormal(const Vec3& p, Vec3* normal) const;
    Trajectory CorrectTrajectory(const Trajectory& traj, double time_offset) const;
    int NumFrames() const { return static_cast<int>(history_.size()); }

   private:
    static constexpr int kPattern = 8;
    static constexpr int kMinPhotoPatches = 20;
    struct PhotoPoint {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::array<Vec3, kPattern> p;
        Eigen::Matrix<double, kPattern, 1> intensity;
        Eigen::Matrix<double, kPattern, 1> weight;
    };
    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id = 0;
        int64_t time_ns = 0;
        LbaState x;
        std::shared_ptr<ImuPreint> imu;  ///< from the previous frame to this one
        cv::Mat gray;                    ///< CV_8U, empty without a camera
        Eigen::aligned_vector<PhotoPoint> photo;  ///< points hosted by this frame
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
    struct History {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int64_t time_ns = 0;
        Eigen::Isometry3d odom = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d refined = Eigen::Isometry3d::Identity();
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
    void SelectPhotoPoints(Frame* frame, const std::vector<LbaPoint>& points) const;
    bool PhotoResidual(const PhotoPoint& pp, const cv::Mat& img, const LbaState& xh, const LbaState& xt,
                       Eigen::Matrix<double, kPattern, 1>* res, Eigen::Matrix<double, kPattern, 12>* J) const;
    double PhotoCost(const Eigen::Matrix<double, kPattern, 1>& res, Eigen::Matrix<double, kPattern, 1>* w) const;
    void GatePhotoPairs(const std::vector<LbaState, Eigen::aligned_allocator<LbaState>>& xs);
    double EvaluatePhoto(const std::vector<LbaState, Eigen::aligned_allocator<LbaState>>& xs, Eigen::MatrixXd* hess,
                         Eigen::VectorXd* jact) const;
    void Optimize();

    LbaParam param_;
    Vec3 g_;  ///< gravity acceleration, world frame
    Eigen::Matrix<double, 6, 6> noise_meas_;
    std::deque<Imu> imu_buf_;
    std::deque<Frame, Eigen::aligned_allocator<Frame>> frames_;
    std::unordered_map<VOXEL_LOCATION, Root> map_;
    std::vector<Root*> active_;
    LbaState newest_;
    CamModel::Ptr cam_;
    Mat3 R_ic_ = Mat3::Identity();
    Vec3 t_ic_ = Vec3::Zero();
    double photo_cap_ = 0.0;
    std::vector<char> photo_pair_;
    int photo_pairs_used_ = 0;
    double photo_inlier_ = 0.0;
    std::vector<const PlaneFactor*> factors_;
    std::vector<History, Eigen::aligned_allocator<History>> history_;
};

}  // namespace faster_lio
#endif  // FASTER_LIO_LOCAL_BA_H
