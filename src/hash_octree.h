#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "common_lib.h"
// ---------------------------------------------------------------------------
// Hash map constants
// ---------------------------------------------------------------------------
#define VOXEL_HASH_P   116101LL
#define VOXEL_MAX_N    10000000000LL

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------
namespace faster_lio {
/// A single point together with its 3×3 world-frame position covariance.
struct PointWithCov
{
    Eigen::Vector3d point;  ///< world-frame 3D position
    Eigen::Matrix3d cov;    ///< 3×3 position covariance
};

/// Result of matching one query point to the map: a point-to-plane correspondence.
struct PlaneMatch
{
    Eigen::Vector3d point_w;                    ///< query point in world frame
    Eigen::Vector3d normal;                     ///< plane normal
    Eigen::Vector3d center;                     ///< plane centroid
    Eigen::Matrix<double, 6, 6> plane_var;      ///< 6×6 plane uncertainty [normal | center]
    double          d;                          ///< plane offset: normal·x + d = 0
    float           dis_to_plane;               ///< signed distance from point to plane
    int             layer;                      ///< octree depth at which the match was found
};

// ---------------------------------------------------------------------------
// Internal plane representation
// ---------------------------------------------------------------------------
struct VoxelPlane
{
    Eigen::Vector3d center   = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal   = Eigen::Vector3d::Zero();
    Eigen::Vector3d y_normal = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_normal = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> plane_var;

    float radius          = 0.f;
    float min_eigen_value = 1.f;
    float mid_eigen_value = 1.f;
    float max_eigen_value = 1.f;
    float d               = 0.f;

    int  points_size = 0;
    bool is_plane    = false;
    bool is_init     = false;

    VoxelPlane()
    {
        covariance = Eigen::Matrix3d::Zero();
        plane_var  = Eigen::Matrix<double, 6, 6>::Zero();
    }
};


// ---------------------------------------------------------------------------
// Octree node
// ---------------------------------------------------------------------------
class OctoTree
{
public:
    OctoTree(int max_layer, int layer,
             int points_size_threshold, int max_points_num,
             float planer_threshold);
    ~OctoTree();

    // Incrementally add one point; re-fits plane when enough points accumulate.
    void Update(const PointWithCov &pv);

    // Internal helpers
    void InitPlane(const std::vector<PointWithCov> &points, VoxelPlane *plane);
    void InitOctoTree();
    void CutOctoTree();

    // Data
    std::vector<PointWithCov> temp_points_;
    VoxelPlane               *plane_ptr_;
    OctoTree                 *leaves_[8];

    double voxel_center_[3];
    float  quater_length_;

    std::vector<int> layer_init_num_;

    int  layer_;
    int  octo_state_;         ///< 0 = leaf (has/fits plane), 1 = branch
    int  max_layer_;
    int  max_points_num_;
    int  points_size_threshold_;
    int  update_size_threshold_;
    int  new_points_;

    float planer_threshold_;

    bool init_octo_;
    bool update_enable_;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct HashOctreeConfig
{
    double           voxel_size       = 0.5;
    int              max_layer        = 2;
    std::vector<int> layer_init_num   = {5, 5, 5, 5, 5}; ///< min points to fit plane per layer
    int              max_points_num   = 50;
    double           planer_threshold = 0.01;   ///< min-eigenvalue threshold for plane detection
    double           sigma_num        = 3.0;    ///< Mahalanobis gate for match acceptance
};

// ---------------------------------------------------------------------------
// Public map interface
// ---------------------------------------------------------------------------
class HashOctree
{
public:
    explicit HashOctree(const HashOctreeConfig &config);
    ~HashOctree();

    /// Insert a batch of world-frame points (with covariance) into the map.
    void AddPoints(const std::vector<PointWithCov> &points);

    /// For each query point find the best-fit plane in the map.
    /// Returns only the points that produced a valid match.
    std::vector<PlaneMatch> Match(const std::vector<PointWithCov> &query_points);

private:
    HashOctreeConfig config_;
    std::unordered_map<VOXEL_LOCATION, OctoTree *> map_;

    VOXEL_LOCATION   ToKey(const Eigen::Vector3d &p) const;
    OctoTree  *GetOrCreateNode(const VOXEL_LOCATION &key);

    void BuildSingleResidual(const PointWithCov &pv,
                             const OctoTree    *octo,
                             int                layer,
                             bool              &success,
                             double            &prob,
                             PlaneMatch        &match);
};
}