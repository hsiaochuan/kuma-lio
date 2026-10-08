#ifndef FASTER_LIO_PGO_TYPES_H
#define FASTER_LIO_PGO_TYPES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Geometry>
#include <memory>


namespace faster_lio {
namespace pgo {

using PgoPoint = pcl::PointXYZI;
using PgoCloud = pcl::PointCloud<PgoPoint>;
using PgoCloudPtr = PgoCloud::Ptr;

/// Dual-pose keyframe.  `pose_odom` is what the front end reported and is
/// never modified; `pose_opt` is the backend estimate.
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<KeyFrame>;

    int id = -1;        ///< dense from 0, doubles as the gtsam::Key
    double time = 0.0;  ///< absolute seconds, same clock as StampedPose::time
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();  ///< T_world_body, front end
    Eigen::Isometry3d pose_opt = Eigen::Isometry3d::Identity();   ///< T_world_body, backend
    PgoCloudPtr cloud;  ///< downsampled, expressed in this keyframe's body frame
};

/// An accepted loop closure, i.e. a BetweenFactor waiting to enter the graph.
struct LoopEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id_from = -1;  ///< historic keyframe
    int id_to = -1;    ///< current keyframe
    Eigen::Isometry3d T_from_to = Eigen::Isometry3d::Identity();
    double fitness = 0.0;  ///< GICP mean squared correspondence distance [m^2]
    double overlap = 0.0;  ///< fraction of source points with a correspondence
    double corr_trans = 0.0;  ///< how far GICP moved the cloud [m]
    double corr_rot = 0.0;    ///< ... and by how much [deg]
};

struct PgoParam {
    bool enable = false;

    // ---- keyframe select ----
    double kf_dis_th = 1.0;        ///< [m]
    double kf_angle_th_deg = 10.0; ///< [deg]
    double kf_max_dt = 0.0;        ///< [s], 0 disables the time-based seal
    double kf_voxel_leaf = 0.4;    ///< [m]
    double kf_max_range = 60.0;    ///< [m], 0 disables the crop
    int kf_min_points = 300;
    int kf_max_staging_points = 500000;  ///< safety valve while stationary

    // ---- loop candidate gating ----
    bool loop_enable = true;
    double loop_search_radius = 25.0;  ///< [m], 2-D (x-y)
    double loop_search_max_dz = 4.0;   ///< [m]
    int loop_min_kf_id = 50;           ///< no loops before this many keyframes exist
    int loop_closest_id_th = 50;       ///< skip the most recent N keyframes
    int loop_kf_gap = 20;              ///< keyframes between two accepted loops
    int loop_max_candidates = 3;       ///< GICP attempts per keyframe

    // ---- submap + GICP verification ----
    int submap_half_size = 40;
    int submap_stride = 4;
    double loop_coarse_leaf = 0.8;      ///< [m]
    double loop_fine_leaf = 0.3;        ///< [m]
    double loop_coarse_max_corr = 5.0;  ///< [m]
    double loop_fine_max_corr = 1.0;    ///< [m]
    int loop_max_iterations = 40;
    double loop_fitness_th = 0.3;       ///< mean squared distance [m^2], lower is better
    double loop_min_overlap = 0.5;      ///< inlier ratio
    double loop_max_corr_trans = 10.0;  ///< [m], gross-error gate
    double loop_max_corr_rot = 30.0;    ///< [deg], gross-error gate
    double loop_min_odom_sigma_ratio = 1.0;

    // ---- graph ----
    double odom_sigma_trans = 0.10;  ///< [m]
    double odom_sigma_rot = 3.0;     ///< [deg]
    double loop_sigma_trans = 0.20;  ///< [m]
    double loop_sigma_rot = 3.0;     ///< [deg]
    double loop_robust_delta = 1.0;  ///< Cauchy
    double prior_sigma_trans = 1e-4;
    double prior_sigma_rot = 1e-4;
    int final_lm_iterations = 50;  ///< cap per LM solve

    bool final_outlier_reject = true;
    double final_outlier_chi2 = 22.46;  ///< 6 dof, 99.9%
    int final_reject_rounds = 3;
    int final_sweep_max_loops = 200;

    // ---- misc ----
    bool save_debug = true;
};

}  // namespace pgo
}  // namespace faster_lio

#endif  // FASTER_LIO_PGO_TYPES_H
