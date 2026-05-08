#include "voxel_map.h"
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tbb/parallel_sort.h>
namespace faster_lio {
VoxelMap::VoxelMap(const Config& config) : config_(config){
}

void VoxelMap::Finish(const double & min_inlier, const double & thr) {
    for (auto & [vox_loc, grid] : voxel_map_) {
        if (!grid || grid->points->empty() || grid->points->size() < min_inlier) {
            continue;
        }

        pcl::SACSegmentation<faster_lio::Point> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(thr);
        seg.setMaxIterations(50);

        pcl::PointIndices inliers;
        pcl::ModelCoefficients coefficients;

        seg.setInputCloud(grid->points);
        seg.segment(inliers, coefficients);

        if (inliers.indices.size() < min_inlier || coefficients.values.size() < 4) {
            grid->plane_coeff.setZero();
            continue;
        }
        for (size_t i = 0; i < inliers.indices.size(); ++i) {
            grid->inliers_.push_back(inliers.indices[i]);
        }
        grid->plane_coeff = Vec4(coefficients.values[0], coefficients.values[1], coefficients.values[2], coefficients.values[3]);
    }
}
bool VoxelMap::AddPoints(const PointCloud::Ptr& points) {
    if (points->empty()) {
        return false;
    }

    std::vector<PointWithVoxLoc> point_buff;
    point_buff.resize(points->size());

    for (size_t i = 0; i < points->size(); ++i) {
        Eigen::Vector3d pw = points->points[i].getVector3fMap().cast<double>();
        PointWithVoxLoc p;
        p.point_ = points->points[i];
        p.vox_loc = VOXEL_LOCATION(pw, config_.voxel_size);
        point_buff[i] = p;
    }

    tbb::parallel_sort(point_buff.begin(), point_buff.end());

    for (size_t i = 0; i < point_buff.size();) {
        size_t j = i;
        VOXEL_LOCATION vox_loc = point_buff.at(i).vox_loc;
        PointCluster cluster_add;
        PointCloud points_add;
        for (; j < point_buff.size() && vox_loc == point_buff.at(j).vox_loc; ++j) {
            Eigen::Vector3d point_j = point_buff.at(j).point_.getVector3fMap().cast<double>();
            cluster_add.Push(point_j);
            points_add.push_back(point_buff.at(j).point_);
        }

        auto iter = voxel_map_.find(vox_loc);
        if (iter == voxel_map_.end()) {
            std::shared_ptr<Grid> grid = std::make_shared<Grid>();
            voxel_map_[vox_loc] = grid;
        }
        voxel_map_[vox_loc]->Update(cluster_add, points_add);
        i = j;
    }

    return true;
}

}  // namespace faster_lio