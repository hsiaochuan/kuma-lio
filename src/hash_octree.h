#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <array>
#include <memory>
#include "common_lib.h"

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

    VoxelPlane()
    {
        covariance = Eigen::Matrix3d::Zero();
        plane_var  = Eigen::Matrix<double, 6, 6>::Zero();
    }
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
    int              update_size_threshold = 5;
    double           planer_threshold = 0.01;   ///< min-eigenvalue threshold for plane detection
    double           sigma_num        = 3.0;    ///< Mahalanobis gate for match acceptance
};

// ---------------------------------------------------------------------------
// Octree node
// ---------------------------------------------------------------------------
class OctoTree : public std::enable_shared_from_this<OctoTree>
{
public:
    OctoTree(int layer);
    ~OctoTree();

    using Ptr = std::shared_ptr<OctoTree>;

    // Initialize shared octree parameters once per map.
    static void SetParams(const HashOctreeConfig &config);

    // Incrementally add one point; re-fits plane when enough points accumulate.
    void Update(const PointWithCov &pv);

    // Internal helpers
    void InitPlane(const std::vector<PointWithCov> &points, VoxelPlane *plane);
    void Fix() {
        update_enable_ = false;
        points_.clear();
        new_cnt    = 0;
    }
    void CutOctoTree();

    // Data
    std::vector<PointWithCov> points_;
    VoxelPlane               *plane_ptr_;
    std::array<Ptr, 8>        leaves_{};

    double voxel_center_[3];
    float  quater_length_;
    int  layer_;
    int  octo_state_ = 0;         ///< 0 = leaf (has/fits plane), 1 = branch
    bool init_octo_ = false;
    int  new_cnt = 0;
    bool update_enable_ = true;

private:
    static HashOctreeConfig config_;
    static int GetLayerInitNum(int layer);

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
    int RemoveVoxelOutOfBounds(const Vec3 &pos_w, double bound_x, double bound_y, double bound_z);

private:
    HashOctreeConfig config_;
    std::unordered_map<VOXEL_LOCATION, OctoTree::Ptr> map_;

    OctoTree::Ptr GetOrCreateNode(const VOXEL_LOCATION &key);

    void BuildSingleResidual(const PointWithCov &pv,
                             const OctoTree::Ptr &octo,
                             int                layer,
                             bool              &success,
                             double            &prob,
                             PlaneMatch        &match);
};
}