#pragma once

#include <glog/logging.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <unordered_map>
#include "common_lib.h"
#include "point_cluster.h"
namespace faster_lio {
struct Grid {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Grid() : normal(Eigen::Vector3d::Zero()) {}
    PointCluster cluster;
    PointCloud points;
    Eigen::Vector3d normal;

    void Update(PointCluster& cluster_add, PointCloud & points_add) {
        // update cluster and normal
        cluster += cluster_add;
        points += points_add;
        normal.setZero();
        if (cluster.N >= 4) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cluster.Cov());
            if (solver.eigenvalues()[0] / solver.eigenvalues()[1] < 0.1) {
                normal = solver.eigenvectors().col(0);
            }
        }
    }
    void Reset() {
        cluster.SetZero();
        normal.setZero();
    }
};

struct PointWithVoxLoc {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PointType point_;
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
    bool AddCloud(const PointCloud::Ptr& input_cloud);
};
}  // namespace faster_lio