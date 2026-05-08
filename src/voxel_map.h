#pragma once

#include <glog/logging.h>


#include <unordered_map>
#include "common_lib.h"
#include "point_cluster.h"
namespace faster_lio {
struct Grid {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Grid() : plane_coeff(Vec4::Zero()), points(new PointCloud) {}
    PointCluster cluster;
    PointCloud::Ptr points;

    // plane coeff
    Vec4 plane_coeff = Vec4::Zero();
    std::vector<int> inliers_;
    void Update(PointCluster& cluster_add, PointCloud & points_add) {
        // update cluster and normal
        cluster += cluster_add;
        *points += points_add;
        plane_coeff.setZero();
    }
    void Reset() {
        cluster.SetZero();
        plane_coeff.setZero();
    }
};

struct PointWithVoxLoc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    faster_lio::Point point_;
    VOXEL_LOCATION vox_loc;
    PointWithVoxLoc() = default;

    bool operator<(const PointWithVoxLoc& p) const { return (vox_loc < p.vox_loc); }
};

class VoxelMap {
   public:
    using Ptr = std::shared_ptr<VoxelMap>;
    struct Config {
        double voxel_size = 0.5;
    };

    explicit VoxelMap(const Config& config);
    Config config_;
    std::unordered_map<VOXEL_LOCATION, std::shared_ptr<Grid>> voxel_map_;
    bool AddPoints(const PointCloud::Ptr& points);
    void Finish(const double & min_inlier, const double & thr);
};
}  // namespace faster_lio