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

#include <Eigen/Geometry>
#include <string>
#include <vector>

#include "eigen_type.h"

namespace kuma_lio {
namespace pgo {

// The candidate layer a loop came from; see PgoParam::btc_enable.
enum class LoopSource {
    kBtc,
    kRadius,
};

// "btc" / "radius", for the logs and the debug dumps.
inline const char* LoopSourceName(LoopSource s) { return s == LoopSource::kBtc ? "btc" : "radius"; }

struct PgoParam {
    // ---- keyframes: Voxel-SLAM seals one every win_size scans ----
    int kf_scans = 10;
    double kf_min_angle = 5.0;  // [deg] the keyframe is skipped below this AND kf_min_dist
    double kf_min_dist = 0.1;   // [m]
    double kf_ds_voxel = 0.2;   // [m] leaf of the stored keyframe cloud

    // ---- plane cells: the loop ICP, the BTC reference planes and the plane overlap all use them ----
    double plane_voxel = 1.0;  // [m]
    int plane_min_points = 10;
    double plane_min_eigen = 0.01;  // [m^2] a cell is a plane below this smallest eigenvalue

    // radius search
    double loop_radius = 20.0;  // [m]
    double loop_max_dz = 4.0;   // [m]
    // [deg] a heading-reversed revisit shares no geometry with a narrow FOV and only starves the ICP,
    // yet ranked by distance it is often the nearest candidate and crowds the matchable ones out
    double loop_max_yaw_diff = 90.0;

    // ---- BTC, defaults from Voxel-SLAM's read_parameters (non-aerial).  The plane cells are the
    // ---- plane_voxel ones above, which Voxel-SLAM's BTC also builds at 1 m / 10 points / 0.01.
    bool btc_enable = true;
    double btc_merge_normal = 0.1;  // |n1 -+ n2| below which two plane cells may merge
    double btc_merge_dist = 0.3;    // [m] and the largest centre-to-plane distance either way
    int btc_ref_planes = 2;         // M, the largest merged planes the cloud is projected onto
    double btc_pixel_res = 0.5;         // [m] pixel of the height-encoded image
    double btc_layer_height = 0.1;  // [m] one bit of the binary descriptor
    double btc_height_min = 0.0;    // [m] height band above the reference plane, at most 64 layers
    double btc_height_max = 5.0;    // [m]
    int btc_min_bits = 10;          // sigma_I, the least intensity a keypoint may have
    bool btc_line_filter = true;    // drop block maxima that sit on a ridge of the image
    double btc_nms_radius = 2.0;    // [m] 3D non-maximum suppression among keypoints
    int btc_max_keypoints = 100;    // strongest keypoints kept per keyframe
    int btc_near_num = 15;          // K, nearest keypoints a triangle is formed from
    double btc_side_len_min = 2.0;      // [m]
    double btc_side_len_max = 50.0;     // [m]
    double btc_side_len_res = 0.2;      // [m] hash quantization of the sides
    double btc_side_tol = 0.01;     // sides match within this fraction of |side|
    double btc_similarity = 0.7;    // sigma_s, mean bit similarity of the three vertices
    int btc_candidates = 20;        // kappa, top-voted keyframes RANSACed
    int btc_min_votes = 5;
    double btc_ransac_dist = 3.0;  // [m] every vertex of an inlier triangle lands within this
    int btc_min_inliers = 4;
    double btc_overlap_normal = 0.2;  // |n1 -+ n2| of overlapping planes, gamma_u
    double btc_overlap_dist = 0.5;    // [m] their point-to-plane distance, gamma_d
    // sigma_p, least plane overlap of the best candidate.  Voxel-SLAM's jud_default; upstream
    // BTC's own icp_threshold (0.15-0.2) is below it and so never decides anything.
    double btc_min_overlap = 0.5;

    // ---- icp_normal ----
    int icp_max_iter = 20;
    // smallest eigenvalue of sum(n n^T), paper appendix B.  Our plane cells are unmerged, so not
    // upstream's 9-15: at 10 Dark02 lost every loop, at 3 it recovered 59% of its ATE
    double icp_eigval = 3.0;
    int icp_min_match = 50;

    // ---- acceptance ----
    int loop_min_kf_gap = 30;       // keyframes between the two ends of a loop
    double loop_min_travel = 30.0;  // [m] path length between them
    double ratio_drift = 0.01;      // drift over travelled distance, paper: at most 1%
    int opt_cooldown_kf = 10;       // keyframes of cooldown between two solves
    // [m] below this a loop is recorded but not solved.  Not the paper's 0.1: it has to clear our
    // plane ICP's own 5-10 cm noise; Hilti exp02 never exceeds 0.32 m and only got worse corrected
    double opt_min_drift = 0.5;

    // ---- noise, as VARIANCES in the gtsam::Pose3 order [rot, trans] ----
    double loop_var = 1e-4;
    double prior_var = 1e-9;
    // Variance of the relative pose between two consecutive scans.  Voxel-SLAM reads this off the
    // local BA Hessian; a TUM file carries no covariance, so it is a constant here.
    double odom_var_rot = 1e-6;    // [rad^2]
    double odom_var_trans = 1e-5;  // [m^2]

    // ---- the coupled front end only (run_mapping_offline --pgo_config); run_pgo ignores them ----
    int map_kfs = 5;              // newest keyframes the local map is rebuilt from after a solve (Voxel-SLAM init_num)
    double reload_radius = 10.0;  // [m] an older keyframe this close to the sensor is loaded into the local map
    double kf_voxel = 0.1;        // [m] leaf of the keyframe clouds kept for the two above (Voxel-SLAM voxel_size / 10)
    // Keep every keyframe's merged cloud at kf_voxel for the two above; set by LoopClosure, not a yaml key.
    bool keep_kf_points = false;
};

// The flat keys of config/pgo.yaml over the defaults above; a key that is absent keeps its default.
void LoadPgoParam(const std::string& file, PgoParam* p);

// A flat `plane_voxel` cell in the keyframe's own body frame: the centre + normal the loop ICP and
// the plane overlap use.  Its point statistics are not kept: only BTC needs them, once, at sealing.
struct PlaneCell {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 center = Vec3::Zero();
    Vec3 normal = Vec3::Zero();
};
using PlaneCells = std::vector<PlaneCell, Eigen::aligned_allocator<PlaneCell>>;

// One scan: a node of the pose graph.
struct ScanNode {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;  // [s]
    // current estimate, T_world_body.  Nothing but a solve ever writes it, so its value on
    // entry to a solve is also what the last solve left, and the per-node correction
    // pose_new * pose_old^-1 is formed exactly once -- no node is ever corrected twice.
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
};

// `kf_scans` consecutive scans merged into the newest one's frame.
struct KeyFrame {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int kf_id = 0;
    int scan_id = 0;          // index of the ScanNode this keyframe rides on
    double travel_len = 0.0;  // travelled distance at this keyframe [m]
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    int points = 0;                  // in the downsampled cloud the planes came from
    PlaneCells planes;               // keyframe body frame, so a pose change costs nothing
    // the merged cloud at kf_voxel, body frame; empty unless PgoParam::keep_kf_points
    std::vector<Eigen::Vector3f> cloud;
};

struct LoopEdge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int from = -1;  // the older keyframe
    int to = -1;    // the current one
    Eigen::Isometry3d T_from_to = Eigen::Isometry3d::Identity();
    double drift = 0.0;                    // [m] how far the loop moves the current pose
    double span = 0.0;                     // [m] travelled between the two keyframes
    double eigval = 0.0;                   // smallest eigenvalue of sum(n n^T) in icp_normal
    LoopSource source = LoopSource::kBtc;  // the layer that found it
};

using ScanNodes = std::vector<ScanNode, Eigen::aligned_allocator<ScanNode>>;
using KeyFrames = std::vector<KeyFrame, Eigen::aligned_allocator<KeyFrame>>;
using LoopEdges = std::vector<LoopEdge, Eigen::aligned_allocator<LoopEdge>>;
using Isometries = std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>;

// A candidate the detector threw away, kept for the debug dump.
struct LoopRejection {
    int from = -1, to = -1;
    std::string reason;
    double value = 0.0;
    int matches = 0;
    int iters = 0;
    double step = 0.0;  // norm of the last translation update [m]
    double nn_median = 0.0;
    int src_n = 0, tar_n = 0;
    double seed_t = 0.0;
    LoopSource source = LoopSource::kBtc;  // the layer that proposed the candidate
};

}  // namespace pgo
}  // namespace kuma_lio
