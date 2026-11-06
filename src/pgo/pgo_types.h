//
// Pose graph optimization after Voxel-SLAM (Liu et al.), section VIII, run offline on the output of
// any odometry: the nodes are the scans, consecutive nodes are tied by the odometry's relative
// pose, and a loop adds one more edge.  Section IX (global mapping, hierarchical global BA) is
// deliberately left out.  Loop candidates come from BTC (btc.h) or, for comparison, from a radius
// search over the keyframe positions.
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

/// Where loop candidates come from; yaml `loop_search: btc | radius`.
enum class LoopSearch {
    kBtc,     ///< BTC descriptors (btc.h), rotation invariant, seeded by the descriptor match
    kRadius,  ///< old keyframes near the current pose estimate, seeded by the odometry
};

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

    // ---- plane cells: the loop ICP, the BTC reference planes and the plane overlap all use them ----
    double plane_voxel = 1.0;      ///< [m]
    int plane_min_points = 10;
    double plane_min_eigen = 0.01;  ///< [m^2] a cell is a plane below this smallest eigenvalue

    // ---- candidates: which old keyframes the loop ICP is tried against ----
    LoopSearch loop_search = LoopSearch::kBtc;
    int loop_min_kf_gap = 30;       ///< keyframes between the two ends of a loop
    double loop_min_travel = 30.0;  ///< [m] path length between them
    // radius search only
    double loop_radius = 20.0;  ///< [m]
    double loop_max_dz = 4.0;   ///< [m]
    int loop_max_candidates = 3;
    /// [deg] skip a candidate whose estimated heading differs by more than this.  With a narrow
    /// FOV the two views of a heading-reversed revisit share no geometry, so such a candidate can
    /// only ever starve the ICP -- but ranked by distance alone it is usually the NEAREST one and
    /// crowds the genuinely matchable candidates out of the shortlist.
    double loop_max_yaw_diff = 90.0;

    // ---- BTC, defaults from Voxel-SLAM's read_parameters (non-aerial).  The plane cells are the
    // ---- plane_voxel ones above, which Voxel-SLAM's BTC also builds at 1 m / 10 points / 0.01.
    double btc_merge_normal = 0.1;  ///< |n1 -+ n2| below which two plane cells may merge
    double btc_merge_dist = 0.3;    ///< [m] and the largest centre-to-plane distance either way
    int btc_ref_planes = 2;         ///< M, the largest merged planes the cloud is projected onto
    double btc_pixel = 0.5;         ///< [m] pixel of the height-encoded image
    double btc_layer_height = 0.1;  ///< [m] one bit of the binary descriptor
    double btc_height_min = 0.0;    ///< [m] height band above the reference plane, at most 64 layers
    double btc_height_max = 5.0;    ///< [m]
    int btc_min_bits = 10;          ///< sigma_I, the least intensity a keypoint may have
    bool btc_line_filter = true;    ///< drop block maxima that sit on a ridge of the image
    double btc_nms_radius = 2.0;    ///< [m] 3D non-maximum suppression among keypoints
    int btc_max_keypoints = 100;    ///< strongest keypoints kept per keyframe
    int btc_near_num = 15;          ///< K, nearest keypoints a triangle is formed from
    double btc_side_min = 2.0;      ///< [m]
    double btc_side_max = 50.0;     ///< [m]
    double btc_side_res = 0.2;      ///< [m] hash quantization of the sides
    double btc_side_tol = 0.01;     ///< sides match within this fraction of |side|
    double btc_similarity = 0.7;    ///< sigma_s, mean bit similarity of the three vertices
    int btc_candidates = 20;        ///< kappa, top-voted keyframes RANSACed
    int btc_min_votes = 5;
    double btc_ransac_dist = 3.0;   ///< [m] every vertex of an inlier triangle lands within this
    int btc_min_inliers = 4;
    double btc_overlap_normal = 0.2;  ///< |n1 -+ n2| of overlapping planes, gamma_u
    double btc_overlap_dist = 0.5;    ///< [m] their point-to-plane distance, gamma_d
    /// sigma_p, least plane overlap of the best candidate.  Voxel-SLAM's jud_default; upstream
    /// BTC's own icp_threshold (0.15-0.2) is below it and so never decides anything.
    double btc_min_overlap = 0.5;

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
    /// Solve at the end whenever a loop is still waiting, however small its drift.  Off, the loops
    /// that never cleared opt_min_drift since the last solve stay out of the result.  On building00
    /// that tail decided between 0.19 and 0.41 m ATE with otherwise equal loop sets.
    bool opt_final_all = false;

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
    int anc_scan_id = 0;    ///< index of the ScanNode this keyframe rides on
    double jour = 0.0;  ///< travelled distance at this keyframe [m]
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    int points = 0;                  ///< in the downsampled cloud the planes came from
    std::vector<PlanePoint> planes;  ///< keyframe body frame, so a pose change costs nothing
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
