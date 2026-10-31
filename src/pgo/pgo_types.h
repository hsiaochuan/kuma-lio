//
// Pose graph optimization after Voxel-SLAM (Liu et al.), section VIII, run offline on the output of
// any odometry: the nodes are the scans, consecutive nodes are tied by the odometry's relative
// pose, and a loop adds one more edge.  Section IX (global mapping, hierarchical global BA) is
// deliberately left out, and BTC is replaced by a radius search over the keyframe positions.
//
// Single session and single threaded.  The scans are replayed in time order and a solve happens on
// the scan that seals a keyframe, so later scans are detected against the already corrected map.
//

#ifndef FASTER_LIO_PGO_TYPES_H
#define FASTER_LIO_PGO_TYPES_H

#include <Eigen/Geometry>
#include <string>
#include <vector>

#include "eigen_type.h"

namespace faster_lio {
namespace pgo {

struct PgoParam {
    // ---- keyframes: Voxel-SLAM seals one every win_size scans ----
    int kf_scans = 10;
    double kf_min_angle = 5.0;  ///< [deg] the keyframe is skipped below this AND kf_min_dist
    double kf_min_dist = 0.1;   ///< [m]
    double kf_ds_voxel = 0.2;   ///< [m] leaf of the stored keyframe cloud
    /// [m] drop returns beyond this before storing the keyframe; 0 disables the crop, which is
    /// the default.  Measured on M3DGR and MCD it never recovered a loop and it cost Dark01 the
    /// only three it had, so it is kept purely as a lever on what the keyframes cost in memory.
    double kf_max_range = 0.0;

    // ---- plane cloud the loop ICP runs on, in place of the BTC plane extraction ----
    double plane_voxel = 1.0;      ///< [m]
    int plane_min_points = 10;
    double plane_min_eigen = 0.01;  ///< [m^2] a cell is a plane below this smallest eigenvalue

    // ---- candidate search ----
    double loop_radius = 20.0;    ///< [m]
    double loop_max_dz = 4.0;     ///< [m]
    int loop_min_kf_gap = 30;     ///< keyframes between the two ends of a loop
    double loop_min_travel = 30.0;  ///< [m] path length between them
    int loop_max_candidates = 3;
    /// [deg] skip a candidate whose estimated heading differs by more than this.  With a narrow
    /// FOV the two views of a heading-reversed revisit share no geometry, so such a candidate can
    /// only ever starve the ICP -- but ranked by distance alone it is usually the NEAREST one and
    /// crowds the genuinely matchable candidates out of the shortlist.
    double loop_max_yaw_diff = 90.0;

    // ---- icp_normal ----
    int icp_max_iter = 20;
    /// smallest eigenvalue of sum(n n^T), the paper's appendix B degeneration criterion.
    /// Upstream runs 9-15 against BTC's merged plane clouds; ours are unmerged and weight the
    /// directions differently, and 10 threw away well-converged alignments on corridor-like
    /// revisits (Dark02 lost every loop at 10 and recovered 59% of its ATE at 3).
    double icp_eigval = 3.0;
    int icp_min_match = 50;

    // ---- acceptance and trigger ----
    double ratio_drift = 0.01;   ///< drift over travelled distance, paper: at most 1%
    int opt_halt_kf = 10;        ///< keyframes of cooldown between two solves
    /// [m] below this a loop is recorded but not solved.  The paper uses 0.1; ours has to clear
    /// the noise floor of our own loop measurement, which is a plane ICP seeded by the odometry
    /// and good to perhaps 5-10 cm, not the 1 cm its edge variance claims.  Measured: every
    /// sequence that gains has corrections of 1.7-3.3 m, while Hilti exp02 -- a confined
    /// multilevel site where the detector fires constantly on short-baseline near-coincidences --
    /// never exceeds 0.32 m and only got worse for being corrected.
    double opt_min_drift = 0.5;
    int isam_extra_iter = 5;

    // ---- noise, as VARIANCES in the gtsam::Pose3 order [rot, trans] ----
    double loop_var = 1e-4;
    double prior_var = 1e-9;
    /// Variance of the relative pose between two consecutive scans.  Voxel-SLAM reads this off the
    /// local BA Hessian; a TUM file carries no covariance, so it is a constant here.
    double odom_var_rot = 1e-6;    ///< [rad^2]
    double odom_var_trans = 1e-5;  ///< [m^2]
};

/// A plane centre and its normal, in the keyframe's own body frame.
struct PlanePoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 p = Vec3::Zero();
    Vec3 n = Vec3::Zero();
};

/// One scan: a node of the pose graph.
struct ScanNode {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;  ///< [s]
    /// current estimate, T_world_body.  Nothing but a solve ever writes it, so its value on
    /// entry to a solve is also what the last solve left, and the per-node correction
    /// pose_new * pose_old^-1 is formed exactly once -- no node is ever corrected twice.
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
};

/// `kf_scans` consecutive scans merged into the newest one's frame.
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = 0;
    int anchor = 0;    ///< index of the ScanNode this keyframe rides on
    double jour = 0.0;  ///< travelled distance at this keyframe [m]
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    std::vector<Vec3> cloud;        ///< downsampled, keyframe body frame
    std::vector<PlanePoint> planes;  ///< body frame as well, so a pose change costs nothing
};

struct LoopEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int from = -1;  ///< the older keyframe
    int to = -1;    ///< the current one
    Eigen::Isometry3d T_from_to = Eigen::Isometry3d::Identity();
    double drift = 0.0;   ///< [m] how far the loop moves the current pose
    double span = 0.0;    ///< [m] travelled between the two keyframes
    double eigval = 0.0;  ///< smallest eigenvalue of sum(n n^T) in icp_normal
};

using ScanNodes = std::vector<ScanNode, Eigen::aligned_allocator<ScanNode>>;
using KeyFrames = std::vector<KeyFrame, Eigen::aligned_allocator<KeyFrame>>;
using LoopEdges = std::vector<LoopEdge, Eigen::aligned_allocator<LoopEdge>>;
using Isometries = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;

/// A candidate the detector threw away, kept for the debug dump.
struct LoopRejection {
    int from = -1, to = -1;
    std::string reason;
    double value = 0.0;
    int matches = 0;
    int iters = 0;
    double step = 0.0;  ///< norm of the last translation update [m]
    double nn_median = 0.0;
    int src_n = 0, tar_n = 0;
    double seed_t = 0.0;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_TYPES_H
