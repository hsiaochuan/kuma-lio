#include "voxel_map.h"
#include <pcl/common/transforms.h>
namespace faster_lio {
VoxelMap::VoxelMap(const Config& config) : config_(config){
}

bool VoxelMap::AddCloud(const PointCloud::Ptr& input_cloud) {
    if (input_cloud->empty()) {
        return false;
    }

    std::vector<PointWithVoxLoc> point_buff;
    point_buff.resize(input_cloud->size());
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        Eigen::Vector3d pw = input_cloud->points[i].getVector3fMap().cast<double>();
        PointWithVoxLoc p;
        p.point_ = input_cloud->points[i];
        p.vox_loc = VOXEL_LOCATION(pw, config_.voxel_size);
        point_buff[i] = p;
    }

    tbb::parallel_sort(point_buff.begin(), point_buff.end());

    for (size_t i = 0; i < point_buff.size();) {
        size_t j = i;
        VOXEL_LOCATION curr_vox_loc = point_buff.at(i).vox_loc;
        PointCluster cluster_add;
        PointCloud points_add;
        for (; j < point_buff.size(); ++j) {
            Eigen::Vector3d point_j = point_buff.at(j).point_.getVector3fMap().cast<double>();
            if (curr_vox_loc == point_buff.at(j).vox_loc) {
                cluster_add.Push(point_j);
                points_add.push_back(point_buff.at(j).point_);
            } else {
                break;
            }
        }

        auto iter = voxel_map_.find(curr_vox_loc);
        if (iter == voxel_map_.end()) {
            std::shared_ptr<Grid> grid = std::make_shared<Grid>();
            voxel_map_[curr_vox_loc] = grid;
        }
        voxel_map_[curr_vox_loc]->Update(cluster_add, points_add);
        i = j;
    }

    return true;
}

}  // namespace faster_lio